from __future__ import annotations

from typing import Protocol


class Embeddings(Protocol):
    """Provider-neutral embeddings contract.

    Matches the interface already consumed by `IndexManager`, the
    `VectorStoreBackend` implementations (Chroma, PGVector), and
    `EmbeddingRouter` — no changes needed downstream.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
