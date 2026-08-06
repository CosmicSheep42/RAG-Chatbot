from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.documents import Document


VectorSearchStrategy = Literal["similarity", "mmr"]


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class VectorStoreBackend(Protocol):
    """Internal vector-store boundary.

    Scores returned by ``search`` are normalized relevance scores:
    0.0 means least relevant and 1.0 means most relevant.
    """

    provider_id: str
    schema_version: str

    def fingerprint_payload(self) -> dict[str, str | int | float | bool | None]:
        ...

    def index_exists(self, namespace: str) -> bool:
        ...

    def replace_documents(self, namespace: str, documents: list[Document]) -> None:
        ...

    def search(
        self,
        namespace: str,
        query: str,
        *,
        k: int,
        fetch_k: int | None = None,
        strategy: VectorSearchStrategy = "similarity",
        lambda_mult: float | None = None,
    ) -> list[ScoredDocument]:
        ...

    def count(self, namespace: str) -> int:
        ...

    def delete_index(self, namespace: str) -> None:
        ...

    def cleanup_namespace_artifacts(self, namespace: str) -> None:
        ...

    def validate_index(self, namespace: str) -> None:
        ...

    def healthcheck(self) -> None:
        ...
