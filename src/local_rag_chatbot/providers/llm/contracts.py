from __future__ import annotations

from typing import Any, Protocol

from local_rag_chatbot.providers.llm.types import ChatResult, ToolSpec


class ChatModel(Protocol):
    """Provider-neutral chat/generation contract.

    Implementations (Ollama, OpenAI, ...) must normalize their provider's
    response shape into `ChatResult`. Callers pass plain message dicts
    (`role`/`content`, with `tool_calls`/`tool_call_id` for tool-calling
    rounds) and optional `ToolSpec`s; no caller should need to know which
    provider produced the result.
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        ...

    async def achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> ChatResult:
        ...
