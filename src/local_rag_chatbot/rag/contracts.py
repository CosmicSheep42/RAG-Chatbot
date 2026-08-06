from __future__ import annotations

from typing import Any, Protocol

from local_rag_chatbot.conversation.conversation_memory import ConversationMessage


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        candidate_limit: int | None = None,
        final_limit: int | None = None,
        dense_k: int | None = None,
        dense_fetch_k: int | None = None,
        sparse_k: int | None = None,
    ) -> list[Any]:
        ...


class ConversationStore(Protocol):
    def get_history(self, conversation_id: str) -> list[ConversationMessage]:
        ...

    def append(self, conversation_id: str, role: str, content: str) -> None:
        ...
