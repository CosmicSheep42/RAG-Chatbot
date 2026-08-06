from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]+$"
CONVERSATION_ID_PATTERN = SESSION_ID_PATTERN


def validate_conversation_id(conversation_id: str | None) -> str | None:
    if conversation_id is None:
        return None
    if not 1 <= len(conversation_id) <= 128:
        raise ValueError("Conversation id must contain 1 to 128 characters.")
    import re

    if re.fullmatch(CONVERSATION_ID_PATTERN, conversation_id) is None:
        raise ValueError("Conversation id may contain only letters, numbers, dot, underscore, colon, and dash.")
    return conversation_id


def validate_session_id(session_id: str | None) -> str | None:
    return validate_conversation_id(session_id)


def resolve_conversation_aliases(
    *,
    conversation_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    normalized_conversation_id = validate_conversation_id(conversation_id)
    normalized_session_id = validate_session_id(session_id)
    if (
        normalized_conversation_id is not None
        and normalized_session_id is not None
        and normalized_conversation_id != normalized_session_id
    ):
        raise ValueError("conversation_id and legacy session_id must match when both are provided.")
    return normalized_conversation_id or normalized_session_id


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=CONVERSATION_ID_PATTERN,
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN,
        description="Legacy alias for conversation_id.",
    )

    @model_validator(mode="after")
    def validate_identifier_aliases(self) -> "ChatRequest":
        resolve_conversation_aliases(
            conversation_id=self.conversation_id,
            session_id=self.session_id,
        )
        return self


class Citation(BaseModel):
    citation_id: str
    source_name: str | None = None
    snippet: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    section: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SourceSnippet(BaseModel):
    source: str
    snippet: str | None = None


class ChatResponse(BaseModel):
    result: str
    sources: list[SourceSnippet] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_calling_used: bool = False
    route: str | None = None
    matched_category: str | None = None
    matched_intent: str | None = None
    score: float | None = None
    reason: str | None = None
    original_query: str | None = None
    standalone_query: str | None = None
    query_rewritten: bool = False


class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None


class StatusResponse(BaseModel):
    ready: bool
    initializing: bool = False
    error: str | None = None
    rag: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    controls: dict[str, Any] = Field(default_factory=dict)
