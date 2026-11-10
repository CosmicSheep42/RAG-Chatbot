from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from local_rag_chatbot.app_errors import AppError
from local_rag_chatbot.config import ChatbotConfig
from local_rag_chatbot.engine import ChatbotEngine
from local_rag_chatbot.observability import configure_logging, metrics, request_id_var
from local_rag_chatbot.api.routes import create_chat_router
from local_rag_chatbot.runtime_controls import AsyncInferenceCapacityLimiter, RuntimeControls, RuntimeControlConfig

logger = logging.getLogger(__name__)


def create_app(
    config: ChatbotConfig | None = None,
    engine: ChatbotEngine | None = None,
    *,
    mount_frontend: bool = True,
    web_dir: Path | str | None = None,
) -> FastAPI:
    if engine is not None and config is None:
        config = engine.config
    if config is None:
        config = ChatbotConfig.from_env()
    if engine is not None and engine.config != config:
        raise ValueError("create_app() received an engine and a different config. Use engine.config or omit config.")
    frontend_dir = Path(web_dir).expanduser() if web_dir is not None else config.web_dir
    if mount_frontend and frontend_dir is not None:
        if not frontend_dir.exists():
            raise ValueError(f"Configured web_dir does not exist: {frontend_dir}")
        if not frontend_dir.is_dir():
            raise ValueError(f"Configured web_dir is not a directory: {frontend_dir}")

    configure_logging(config.log_level)
    engine = engine or ChatbotEngine(config=config)
    runtime_controls = RuntimeControls(RuntimeControlConfig(**config.runtime_control_config()))
    capacity_limiter = AsyncInferenceCapacityLimiter(
        max_active=config.max_concurrent_inferences,
        max_pending=config.max_pending_inferences,
        retry_after_seconds=min(max(config.inference_timeout_seconds, 1.0), 30.0),
    )
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        try:
            await engine.initialize()
            yield
        finally:
            await engine.shutdown()

    app = FastAPI(title="Local RAG Chatbot", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, _exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            {"error": "Invalid request.", "request_id": request_id},
            status_code=400,
        )

    @app.middleware("http")
    async def request_observability_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except AppError as exc:
            status_code = exc.status_code
            retry_after = getattr(exc, "retry_after", None)
            headers = {"Retry-After": str(max(1, int(retry_after)))} if retry_after is not None else None
            response = JSONResponse(
                {"error": exc.public_message, "request_id": request_id},
                status_code=exc.status_code,
                headers=headers,
            )
            return response
        except Exception:
            logger.exception("Unhandled request error")
            response = JSONResponse({"error": "Internal server error.", "request_id": request_id}, status_code=500)
            return response
        finally:
            elapsed = time.perf_counter() - started
            metrics.increment("http_requests_total", method=request.method, route=request.url.path, status=status_code)
            metrics.observe_latency("http_request_total_seconds", elapsed, route=request.url.path)
            if response is not None:
                response.headers["X-Request-Id"] = request_id
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["Referrer-Policy"] = "no-referrer"
                response.headers["Content-Security-Policy"] = (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "object-src 'none'; frame-ancestors 'none'"
                )
            request_id_var.reset(token)

    app.include_router(
        create_chat_router(
            engine,
            runtime_controls=runtime_controls,
            capacity_limiter=capacity_limiter,
            request_timeout_seconds=config.request_timeout_seconds,
            inference_timeout_seconds=config.inference_timeout_seconds,
        ),
        prefix="/api/v1",
    )
    app.include_router(
        create_chat_router(
            engine,
            runtime_controls=runtime_controls,
            capacity_limiter=capacity_limiter,
            request_timeout_seconds=config.request_timeout_seconds,
            inference_timeout_seconds=config.inference_timeout_seconds,
        ),
        prefix="/api",
        include_in_schema=False,
    )

    if mount_frontend and frontend_dir is not None:
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(frontend_dir / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def api_root():
            return {"status": "ok", "docs": "/docs"}

    return app
