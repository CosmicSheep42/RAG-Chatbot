from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_chatbot.app_errors import ConfigurationError
from local_rag_chatbot.providers.embeddings.contracts import Embeddings

if TYPE_CHECKING:
    from local_rag_chatbot.settings import AppSettings


def create_embeddings(settings: AppSettings) -> Embeddings:
    if settings.embedding_provider == "ollama":
        from local_rag_chatbot.providers.embeddings.ollama import create_ollama_embeddings

        return create_ollama_embeddings(settings)
    if settings.embedding_provider == "openai":
        from local_rag_chatbot.providers.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
