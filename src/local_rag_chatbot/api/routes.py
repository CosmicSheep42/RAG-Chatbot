from __future__ import annotations

import asyncio
import inspect
import warnings

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from local_rag_chatbot.api.auth import (
    AuthProvider,
    ConversationResolver,
    SessionResolver,
    resolve_request_conversation_id,
)
from local_rag_chatbot.engine import ChatbotEngine
from local_rag_chatbot.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    StatusResponse,
    validate_conversation_id,
)
from local_rag_chatbot.observability import metrics
from local_rag_chatbot.request_context import RequestContext
from local_rag_chatbot.runtime_controls import AsyncInferenceCapacityLimiter, CapacityLimitExceeded, RuntimeControls


def create_chat_router(
    engine: ChatbotEngine,
    *,
    auth_provider: AuthProvider | None = None,
    conversation_resolver: ConversationResolver | None = None,
    session_resolver: SessionResolver | None = None,
    runtime_controls: RuntimeControls | None = None,
    capacity_limiter: AsyncInferenceCapacityLimiter | None = None,
    request_timeout_seconds: float | None = None,
    inference_timeout_seconds: float | None = None,
) -> APIRouter:
    router = APIRouter()
    if auth_provider is not None:
        warnings.warn(
            "create_chat_router(auth_provider=...) is deprecated. Protect embedded routes with host FastAPI "
            "dependencies instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if session_resolver is not None:
        warnings.warn(
            "create_chat_router(session_resolver=...) is deprecated; use conversation_resolver=...",
            DeprecationWarning,
            stacklevel=2,
        )

    @router.get(
        "/status",
        response_model=StatusResponse,
        responses={401: {"model": ErrorResponse}},
    )
    async def status(request: Request):
        if auth_provider is not None:
            auth_provider.require_request(request)
        payload = engine.status()
        if capacity_limiter is not None:
            payload["controls"] = {
                **payload.get("controls", {}),
                "active_inferences": capacity_limiter.active,
                "pending_inferences": capacity_limiter.pending,
            }
        return payload

    @router.get("/health", responses={401: {"model": ErrorResponse}})
    async def health(request: Request):
        if auth_provider is not None:
            auth_provider.require_request(request, health=True)
        return {"status": "ok"}

    @router.get("/ready", responses={503: {"model": StatusResponse}, 401: {"model": ErrorResponse}})
    async def ready(request: Request):
        if auth_provider is not None:
            auth_provider.require_request(request, health=True)
        payload = engine.status()
        if payload.get("initializing") or payload.get("error") or not payload.get("ready"):
            return JSONResponse(payload, status_code=503)
        return payload

    @router.get("/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request):
        if auth_provider is not None:
            auth_provider.require_request(request, health=True)
        rag = engine.status().get("rag") or {}
        metrics.set_gauge("rag_index_documents", rag.get("document_count", 0) or 0)
        return PlainTextResponse(metrics.prometheus_text(), media_type="text/plain; version=0.0.4")

    @router.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def chat(payload: ChatRequest, request: Request):
        if auth_provider is not None:
            auth_provider.require_request(request)
        try:
            conversation_id = resolve_request_conversation_id(request, payload)
            if session_resolver is not None:
                legacy_id = session_resolver(request, payload.session_id)
                if conversation_id is not None and legacy_id is not None and conversation_id != legacy_id:
                    raise ValueError("Conflicting conversation identifiers from request and legacy session resolver.")
                conversation_id = conversation_id or legacy_id
            if conversation_resolver is not None:
                resolved = conversation_resolver(request, payload)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                conversation_id = validate_conversation_id(resolved)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        controls = runtime_controls
        if controls is not None:
            allowed, message = controls.validate_input(payload.query)
            if not allowed:
                return JSONResponse({"error": message or "Invalid message."}, status_code=400)
            allowed, retry_after = controls.allow_request()
            if not allowed:
                return JSONResponse(
                    {"error": "Rate limit exceeded."},
                    status_code=429,
                    headers={"Retry-After": str(max(1, int(retry_after)))},
                )
        elif len(payload.query) > engine.config.max_input_chars:
            return JSONResponse(
                {"error": f"Message exceeds the maximum size of {engine.config.max_input_chars} characters."},
                status_code=400,
            )
        if not engine.ready:
            return JSONResponse({"error": "Chatbot engine is not ready."}, status_code=503)
        try:
            async def execute_chat() -> ChatResponse:
                _ = inference_timeout_seconds
                return await engine.chat(
                    payload.query,
                    conversation_id=conversation_id,
                    request_context=RequestContext(
                        request_id=getattr(request.state, "request_id", ""),
                        conversation_id=conversation_id,
                    ),
                )

            async def execute_with_capacity() -> ChatResponse:
                if capacity_limiter is None:
                    return await execute_chat()
                async with capacity_limiter.slot():
                    return await execute_chat()

            if request_timeout_seconds is None:
                return await execute_with_capacity()
            async with asyncio.timeout(request_timeout_seconds):
                return await execute_with_capacity()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except CapacityLimitExceeded as exc:
            return JSONResponse(
                {"error": "Inference capacity is saturated."},
                status_code=503,
                headers={"Retry-After": str(max(1, int(exc.retry_after)))},
            )
        except TimeoutError:
            return JSONResponse({"error": "Request timed out."}, status_code=504)
        except Exception:
            return JSONResponse({"error": "Error processing query."}, status_code=500)

    return router
