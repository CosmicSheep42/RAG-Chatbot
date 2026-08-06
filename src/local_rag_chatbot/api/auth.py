from __future__ import annotations

from typing import Awaitable, Protocol
import warnings

from fastapi import Request

from local_rag_chatbot.models import resolve_conversation_aliases, validate_conversation_id


class AuthProvider(Protocol):
    def require_request(self, request: Request, *, health: bool = False) -> str | None:
        ...


class SessionResolver(Protocol):
    def __call__(self, request: Request, body_session_id: str | None = None) -> str | None:
        ...


class ConversationResolver(Protocol):
    def __call__(self, request: Request, payload: object) -> str | None | Awaitable[str | None]:
        ...


def default_session_resolver(request: Request, body_session_id: str | None = None) -> str | None:
    warnings.warn(
        "default_session_resolver() is deprecated; use resolve_request_conversation_id().",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_conversation_aliases(
        conversation_id=None,
        session_id=body_session_id or request.headers.get("x-session-id"),
    )


def resolve_request_conversation_id(request: Request, payload: object) -> str | None:
    candidates = [
        ("body conversation_id", getattr(payload, "conversation_id", None)),
        ("body session_id", getattr(payload, "session_id", None)),
        ("X-Conversation-Id", request.headers.get("x-conversation-id")),
        ("X-Session-Id", request.headers.get("x-session-id")),
    ]
    resolved: str | None = None
    resolved_label: str | None = None
    for label, raw_value in candidates:
        value = validate_conversation_id(raw_value)
        if value is None:
            continue
        if resolved is not None and value != resolved:
            raise ValueError(f"Conflicting conversation identifiers: {resolved_label} and {label}.")
        resolved = value
        resolved_label = label
    return resolved
