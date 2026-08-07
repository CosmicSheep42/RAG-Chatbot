from __future__ import annotations

from local_rag_chatbot.providers.embeddings.contracts import Embeddings
from local_rag_chatbot.providers.embeddings.factory import create_embeddings

__all__ = ["Embeddings", "create_embeddings"]
