from __future__ import annotations

from typing import TYPE_CHECKING

from local_rag_chatbot.providers.embeddings.contracts import Embeddings

if TYPE_CHECKING:
    from local_rag_chatbot.settings import AppSettings


def create_ollama_embeddings(settings: AppSettings) -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
