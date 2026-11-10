from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import logging
import math
import shutil
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document

from local_rag_chatbot.vectorstores.contracts import ScoredDocument, VectorSearchStrategy

logger = logging.getLogger(__name__)
CHROMA_DEPENDENCY_ERROR = (
    "Chroma backend dependencies are not installed.\n"
    'Install them with: pip install -e ".[chroma]"'
)

try:
    import chromadb  # noqa: F401
    from langchain_community.vectorstores import Chroma
except ModuleNotFoundError:  # pragma: no cover - depends on selected optional extra
    Chroma = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ChromaVectorStoreSettings:
    rag_persist_directory: Path
    router_persist_directory: Path


class ChromaVectorStoreBackend:
    provider_id = "chroma"
    schema_version = "chroma-local-v1"
    rag_namespace = "rag"

    def __init__(self, *, settings: ChromaVectorStoreSettings, embeddings: object) -> None:
        self.settings = settings
        self.embeddings = embeddings
        if Chroma is None:
            from local_rag_chatbot.app_errors import ConfigurationError

            raise ConfigurationError(CHROMA_DEPENDENCY_ERROR)

    def fingerprint_payload(self) -> dict[str, str | int | float | bool | None]:
        return {
            "provider": self.provider_id,
            "schema_version": self.schema_version,
            "persistence_mode": "local_directory",
            "distance_metric": "backend_default",
        }

    def index_exists(self, namespace: str) -> bool:
        directory = self._namespace_directory(namespace)
        return directory.exists() and (directory / "chroma.sqlite3").exists()

    def replace_documents(self, namespace: str, documents: list[Document]) -> None:
        if not documents:
            raise ValueError(f"No documents supplied for vector namespace '{namespace}'.")
        directory = self._namespace_directory(namespace)
        staging_directory = directory.with_name(f"{directory.name}.{uuid4().hex}.staging")
        backup_directory = directory.with_name(f"{directory.name}.backup")
        self._cleanup_abandoned_artifacts(staging_directory, backup_directory, directory)
        self._build_store(staging_directory, namespace, documents)
        self._validate_directory(staging_directory, namespace)

        moved_current = False
        try:
            if directory.exists():
                directory.replace(backup_directory)
                moved_current = True
            staging_directory.replace(directory)
            self._reset_chroma_client_cache()
            self._validate_directory(directory, namespace)
            if backup_directory.exists():
                shutil.rmtree(backup_directory)
        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory, ignore_errors=True)
            if moved_current and backup_directory.exists() and not directory.exists():
                backup_directory.replace(directory)
                self._reset_chroma_client_cache()
            raise

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
        store = self._load_store(namespace)
        if strategy == "mmr":
            retriever = store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k or k,
                    **({"lambda_mult": lambda_mult} if lambda_mult is not None else {}),
                },
            )
            documents = list(retriever.invoke(query))
            return [
                # MMR exposes selection order, not a comparable relevance score.
                ScoredDocument(self._with_score(document, 0.0), 0.0)
                for document in documents
            ]
        results = store.similarity_search_with_relevance_scores(query, k=k)
        return [
            ScoredDocument(self._with_score(document, self._normalize_relevance_score(score)), self._normalize_relevance_score(score))
            for document, score in results
        ]

    def count(self, namespace: str) -> int:
        store = self._load_store(namespace)
        return self._count_store(store)

    def _count_store(self, store: Chroma) -> int:
        collection = getattr(store, "_collection", None)
        # LangChain's Chroma wrapper does not expose count publicly; keep the private call contained here.
        if collection is None:
            return 0
        return int(collection.count())

    def delete_index(self, namespace: str) -> None:
        directory = self._namespace_directory(namespace)
        if directory.exists():
            shutil.rmtree(directory)
            self._reset_chroma_client_cache()

    def healthcheck(self) -> None:
        self.settings.rag_persist_directory.parent.mkdir(parents=True, exist_ok=True)
        self.settings.router_persist_directory.mkdir(parents=True, exist_ok=True)

    def cleanup_namespace_artifacts(self, namespace: str) -> None:
        directory = self._namespace_directory(namespace)
        for staging_directory in directory.parent.glob(f"{directory.name}.*.staging"):
            logger.warning("Removing abandoned vector index staging directory", extra={"path": str(staging_directory)})
            shutil.rmtree(staging_directory, ignore_errors=True)
        self._cleanup_abandoned_artifacts(
            directory.with_name(f"{directory.name}.staging"),
            directory.with_name(f"{directory.name}.backup"),
            directory,
        )

    def validate_index(self, namespace: str) -> None:
        self._validate_directory(self._namespace_directory(namespace), namespace)

    def _build_store(self, directory: Path, namespace: str, documents: list[Document]) -> Chroma:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Creating Chroma vector store", extra={"namespace": namespace, "document_count": len(documents)})
        return Chroma.from_documents(
            documents=[self._document_for_chroma(document) for document in documents],
            embedding=self.embeddings,
            persist_directory=str(directory),
            collection_name=self._collection_name(namespace),
        )

    def _load_store(self, namespace: str) -> Chroma:
        return Chroma(
            persist_directory=str(self._namespace_directory(namespace)),
            collection_name=self._collection_name(namespace),
            embedding_function=self.embeddings,
        )

    def _validate_directory(self, directory: Path, namespace: str) -> None:
        if not (directory.exists() and (directory / "chroma.sqlite3").exists()):
            raise RuntimeError(f"Vector store was not created for namespace '{namespace}'.")
        store = Chroma(
            persist_directory=str(directory),
            collection_name=self._collection_name(namespace),
            embedding_function=self.embeddings,
        )
        if self._count_store(store) <= 0:
            raise RuntimeError(f"Vector store validation failed for namespace '{namespace}': no documents indexed.")

    def _cleanup_abandoned_artifacts(self, staging_directory: Path, backup_directory: Path, directory: Path) -> None:
        if staging_directory.exists():
            logger.warning("Removing abandoned vector index staging directory", extra={"path": str(staging_directory)})
            shutil.rmtree(staging_directory)
        if backup_directory.exists() and directory.exists():
            logger.warning("Removing stale vector index backup directory", extra={"path": str(backup_directory)})
            shutil.rmtree(backup_directory)
        elif backup_directory.exists() and not directory.exists():
            logger.warning("Restoring vector index backup because active directory is missing", extra={"path": str(backup_directory)})
            backup_directory.replace(directory)

    def _namespace_directory(self, namespace: str) -> Path:
        if namespace == self.rag_namespace:
            return self.settings.rag_persist_directory
        if namespace.startswith("router:"):
            return self.settings.router_persist_directory / namespace.split(":", 1)[1]
        return self.settings.rag_persist_directory.parent / self._safe_namespace(namespace)

    def _collection_name(self, namespace: str) -> str:
        if namespace == self.rag_namespace:
            return "langchain"
        return self._safe_namespace(namespace)

    def _safe_namespace(self, namespace: str) -> str:
        return namespace.replace(":", "_").replace("/", "_").replace("\\", "_")

    def _document_for_chroma(self, document: Document) -> Document:
        vector_document = copy.copy(document)
        vector_document.metadata = self._metadata_for_chroma(getattr(document, "metadata", {}) or {})
        return vector_document

    def _metadata_for_chroma(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        chroma_metadata: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                chroma_metadata[key] = value
            elif value is None:
                continue
            else:
                chroma_metadata[key] = "__local_rag_json__" + json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        return chroma_metadata

    def _normalize_relevance_score(self, score: float) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    def _with_score(self, document: Document, score: float) -> Document:
        metadata: dict[str, Any] = {}
        for key, value in document.metadata.items():
            if isinstance(value, str) and value.startswith("__local_rag_json__"):
                try:
                    metadata[key] = json.loads(value.removeprefix("__local_rag_json__"))
                    continue
                except json.JSONDecodeError:
                    pass
            metadata[key] = value
        cloned = Document(page_content=document.page_content, metadata=metadata)
        cloned.metadata["retrieval_dense_score"] = round(score, 6)
        return cloned

    def _reset_chroma_client_cache(self) -> None:
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            logger.debug("Could not clear Chroma client cache after directory lifecycle operation", exc_info=True)
