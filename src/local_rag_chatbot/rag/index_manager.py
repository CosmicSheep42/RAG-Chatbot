from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import os
import tempfile
from typing import Any, TYPE_CHECKING

from local_rag_chatbot.documents.hash_manager import HashManager
from local_rag_chatbot.documents.index_manifest import IndexManifest, ManifestEntry
from local_rag_chatbot.file_lock import FileLock, FileLockTimeoutError

if TYPE_CHECKING:
    from local_rag_chatbot.documents.document_repository import DocumentRepository
    from local_rag_chatbot.documents.document_splitter import DocumentSplitter
    from local_rag_chatbot.rag.context_assembler import DocumentContextAssembler
    from local_rag_chatbot.rag.hybrid_retriever import HybridRetriever, RetrievalTuning
    from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceEstimator
    from local_rag_chatbot.rag.retrieval_service import RetrievalService
    from local_rag_chatbot.settings import AppSettings
    from local_rag_chatbot.vectorstores import VectorStoreBackend

logger = logging.getLogger(__name__)
PARSER_VERSION = "loaders-v2"
CHUNKER_VERSION = "structural-v2"
INDEX_LOCK_TIMEOUT_SECONDS = 60.0
TRANSACTION_VERSION = 1
RAG_VECTOR_NAMESPACE = "rag"


class RAGIndexStatus(str, Enum):
    NOT_INITIALIZED = "not_initialized"
    EMPTY = "empty"
    INDEXED = "indexed"
    ERROR = "error"


@dataclass(frozen=True)
class IndexLoadResult:
    initialized: bool
    status: dict[str, Any]


class IndexManager:
    def __init__(
        self,
        *,
        settings: AppSettings,
        document_repository: DocumentRepository,
        splitter: DocumentSplitter,
        hash_manager: HashManager,
        manifest: IndexManifest,
        embeddings: Any,
        vector_backend: VectorStoreBackend,
        retrieval_tuning: RetrievalTuning,
        confidence_estimator: RetrievalConfidenceEstimator | None = None,
    ) -> None:
        self.settings = settings
        self.document_repository = document_repository
        self.splitter = splitter
        self.hash_manager = hash_manager
        self.manifest = manifest
        self.embeddings = embeddings
        self.vector_backend = vector_backend
        self.retrieval_tuning = retrieval_tuning
        if confidence_estimator is None:
            from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceEstimator

            confidence_estimator = RetrievalConfidenceEstimator()
        self.confidence_estimator = confidence_estimator
        self.documents: list[Any] = []
        self.retriever: HybridRetriever | None = None
        self.context_assembler: DocumentContextAssembler | None = None
        self.retrieval_service: RetrievalService | None = None
        self.rag_status = RAGIndexStatus.NOT_INITIALIZED
        self.rag_error: str | None = None

    @property
    def has_indexed_documents(self) -> bool:
        return self.rag_status == RAGIndexStatus.INDEXED and self.retriever is not None

    @property
    def has_empty_corpus(self) -> bool:
        return self.rag_status == RAGIndexStatus.EMPTY

    def status(self) -> dict[str, Any]:
        return {
            "status": self.rag_status.value,
            "available": self.has_indexed_documents,
            "empty_corpus": self.has_empty_corpus,
            "document_count": len(self.documents),
            "error": self.rag_error,
        }

    def load_existing(self) -> bool:
        try:
            with FileLock(self.settings.data_dir / "rag_index.lock", timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
                self._recover_index_transaction()
            splits = self._load_documents()
            if not splits:
                logger.info("No documents available for processing")
                self._mark_empty_corpus()
                return False

            discovered = self._manifest_from_documents(splits)
            if (
                not self.manifest.corrupt
                and not self.manifest.has_incompatible_changes(discovered)
                and not any(self.manifest.classify(discovered)[key] for key in ("added", "modified", "deleted"))
            ):
                logger.info("Loading existing vector store from matching manifest")
                if not self._load_vectorstore():
                    return False
                self._build_retrieval_stack(splits, update_manifest=False)
                self._mark_indexed()
                logger.info("Vector store loaded")
                return True

            logger.info("Existing vector store needs safe rebuild")
            return False
        except Exception as exc:
            self._mark_error(exc)
            raise

    def prepare(self) -> bool:
        try:
            splits = self._load_documents()
            if not splits:
                self._mark_empty_corpus()
                return True

            discovered = self._manifest_from_documents(splits)
            current_hash = self.hash_manager.compute_directory_hash(
                self.settings.documents_dir,
                patterns=[
                    f"*{extension}"
                    for extension in self.document_repository.loader_registry.supported_extensions
                ],
                ignored_paths=self.document_repository.ignored_paths,
                ignored_directory_paths=self.document_repository.ignored_directory_paths,
            )

            self._build_or_update_vectorstore_safely(splits, discovered)
            self._build_retrieval_stack(splits, update_manifest=False)
            self.hash_manager.save_hash("rag_documents", current_hash)
            self._mark_indexed()
            logger.info("RAG configured successfully")
            return True
        except Exception as exc:
            self._mark_error(exc)
            raise

    def load_or_prepare(self, *, auto_prepare: bool = True) -> bool:
        if self.load_existing():
            return True
        if not auto_prepare:
            self.rag_error = (
                "RAG index is missing or incompatible. Run 'python -m local_rag_chatbot.ingest' "
                "or enable AUTO_INDEX_ON_STARTUP=true."
            )
            if self.rag_status != RAGIndexStatus.EMPTY:
                self.rag_status = RAGIndexStatus.ERROR
            return False
        return self.prepare()

    def _mark_empty_corpus(self) -> None:
        self.documents = []
        self.retriever = None
        self.context_assembler = None
        self.retrieval_service = None
        self.rag_status = RAGIndexStatus.EMPTY
        self.rag_error = None

    def _mark_indexed(self) -> None:
        self.rag_status = RAGIndexStatus.INDEXED
        self.rag_error = None

    def _mark_error(self, exc: Exception) -> None:
        self.rag_status = RAGIndexStatus.ERROR
        self.rag_error = str(exc)

    def _load_documents(self) -> list[Any]:
        documents = self.document_repository.load()
        if not documents:
            return []
        splits = self.splitter.split(documents)
        logger.info("Created RAG chunks", extra={"chunk_count": len(splits)})
        return splits

    def _load_vectorstore(self) -> bool:
        if not self.vector_backend.index_exists(RAG_VECTOR_NAMESPACE):
            return False
        self.vector_backend.healthcheck()
        return True

    def _build_retrieval_stack(self, splits: list[Any], *, update_manifest: bool = True) -> None:
        if not self.vector_backend.index_exists(RAG_VECTOR_NAMESPACE):
            raise RuntimeError("Vector store is not initialized.")
        from local_rag_chatbot.rag.context_assembler import DocumentContextAssembler
        from local_rag_chatbot.rag.hybrid_retriever import HybridRetriever
        from local_rag_chatbot.rag.retrieval_service import RetrievalService

        self.documents = list(splits)
        self.context_assembler = DocumentContextAssembler(self.documents)
        self.retriever = HybridRetriever(
            vector_backend=self.vector_backend,
            namespace=RAG_VECTOR_NAMESPACE,
            documents=self.documents,
            tuning=self.retrieval_tuning,
            reranking_mode=self.settings.reranking_mode,
            reranker_model=self.settings.reranker_model,
            reranker_allow_lexical_fallback=self.settings.reranker_allow_lexical_fallback,
        )
        self.retrieval_service = RetrievalService(
            retriever=self.retriever,
            context_assembler=self.context_assembler,
            confidence_estimator=self.confidence_estimator,
        )
        if update_manifest:
            self._save_manifest(self._manifest_from_documents(self.documents))

    def _manifest_from_documents(self, documents: list[Any]) -> dict[str, ManifestEntry]:
        by_source: dict[str, list[Any]] = {}
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            source = metadata.get("source")
            if source:
                by_source.setdefault(source, []).append(document)

        discovered: dict[str, ManifestEntry] = {}
        for source, chunks in by_source.items():
            path = self.settings.documents_dir / source
            first_metadata = getattr(chunks[0], "metadata", {}) or {}
            try:
                file_hash = str(first_metadata.get("document_hash") or self._file_hash(path))
                file_type = path.suffix.lower().lstrip(".")
            except OSError:
                continue
            discovered[source] = ManifestEntry(
                document_id=str(first_metadata.get("document_id", "")),
                relative_path=source,
                file_hash=file_hash,
                file_type=file_type,
                parser_version=PARSER_VERSION,
                chunker_version=self._chunker_signature(),
                embedding_model=self.settings.embedding_model,
                embedding_config=self._embedding_signature(),
                indexed_chunk_ids=[str((getattr(chunk, "metadata", {}) or {}).get("chunk_id", "")) for chunk in chunks],
                indexed_at=datetime.now(timezone.utc).isoformat(),
            )
        return discovered

    def _chunker_signature(self) -> str:
        return json.dumps(
            {
                "version": CHUNKER_VERSION,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            },
            sort_keys=True,
        )

    def _embedding_signature(self) -> str:
        return json.dumps(
            {
                "backend_provider": self.vector_backend.provider_id,
                "backend_schema_version": self.vector_backend.schema_version,
                "backend": self.vector_backend.fingerprint_payload(),
                "embedding_provider": self._embedding_provider_identifier(),
                "embedding_model": self._embedding_model_identifier(),
                "embedding_dimensions": self._embedding_dimensions(),
                "embedding_normalization": "provider_default",
            },
            sort_keys=True,
        )

    def _embedding_provider_identifier(self) -> str:
        provider_type = type(self.embeddings)
        return f"{provider_type.__module__}.{provider_type.__qualname__}"

    def _embedding_model_identifier(self) -> str:
        for attribute in ("model", "model_name", "embedding_model"):
            value = getattr(self.embeddings, attribute, None)
            if value:
                return str(value)
        return self.settings.embedding_model

    def _embedding_dimensions(self) -> int | None:
        for attribute in ("dimensions", "embedding_dimensions"):
            value = getattr(self.embeddings, attribute, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _save_manifest(self, discovered: dict[str, ManifestEntry]) -> None:
        changes = self.manifest.classify(discovered)
        logger.info("Index manifest compared", extra={f"manifest_{key}": len(value) for key, value in changes.items()})
        self.manifest.entries = discovered
        self.manifest.corrupt = False
        self.manifest.save()

    def _update_manifest(self, documents: list[Any]) -> None:
        self._save_manifest(self._manifest_from_documents(documents))

    def _chunks_for_paths(self, documents: list[Any], paths: set[str]) -> list[Any]:
        return [
            document
            for document in documents
            if str((getattr(document, "metadata", {}) or {}).get("source", "")) in paths
        ]

    def _chunk_ids_for_paths(self, paths: list[str]) -> list[str]:
        chunk_ids: list[str] = []
        for path in paths:
            entry = self.manifest.entries.get(path)
            if entry:
                chunk_ids.extend(chunk_id for chunk_id in entry.indexed_chunk_ids if chunk_id)
        return chunk_ids

    def _replace_vectorstore_atomically(self, splits: list[Any]) -> None:
        self._replace_index_and_manifest_atomically(splits, self._manifest_from_documents(splits))

    def _replace_index_and_manifest_atomically(
        self,
        splits: list[Any],
        discovered: dict[str, ManifestEntry],
    ) -> None:
        manifest_path = self.manifest.path
        staging_manifest = manifest_path.with_name(f"{manifest_path.name}.staging")
        backup_manifest = manifest_path.with_name(f"{manifest_path.name}.backup")
        journal_path = self._transaction_journal_path()

        self._recover_index_transaction()
        self._cleanup_abandoned_index_artifacts(staging_manifest, backup_manifest, manifest_path)

        self.vector_backend.replace_documents(RAG_VECTOR_NAMESPACE, splits)
        self.vector_backend.validate_index(RAG_VECTOR_NAMESPACE)
        self._write_manifest(staging_manifest, discovered)
        self._validate_manifest_file(staging_manifest)

        try:
            self._write_transaction_journal(
                "prepared",
                active_manifest=manifest_path,
                staging_manifest=staging_manifest,
                backup_manifest=backup_manifest,
            )
            if manifest_path.exists():
                manifest_path.replace(backup_manifest)
            staging_manifest.replace(manifest_path)
            self._write_transaction_journal(
                "new_promoted",
                active_manifest=manifest_path,
                staging_manifest=staging_manifest,
                backup_manifest=backup_manifest,
            )
            self.vector_backend.healthcheck()
            self._validate_manifest_file(manifest_path)
            self._write_transaction_journal(
                "validated",
                active_manifest=manifest_path,
                staging_manifest=staging_manifest,
                backup_manifest=backup_manifest,
            )
            if backup_manifest.exists():
                backup_manifest.unlink()
            if journal_path.exists():
                journal_path.unlink()
        except Exception:
            self._recover_index_transaction()
            raise

        self.manifest.entries = discovered
        self.manifest.corrupt = False

    def _build_or_update_vectorstore_safely(
        self,
        splits: list[Any],
        discovered: dict[str, ManifestEntry],
        *,
        force_rebuild: bool = False,
    ) -> None:
        lock_path = self.settings.data_dir / "rag_index.lock"
        try:
            with FileLock(lock_path, timeout_seconds=INDEX_LOCK_TIMEOUT_SECONDS):
                self._recover_index_transaction()
                self._cleanup_abandoned_index_artifacts(
                    self.manifest.path.with_name(f"{self.manifest.path.name}.staging"),
                    self.manifest.path.with_name(f"{self.manifest.path.name}.backup"),
                    self.manifest.path,
                )
                self.vector_backend.cleanup_namespace_artifacts(RAG_VECTOR_NAMESPACE)
                needs_rebuild = force_rebuild or self.manifest.has_incompatible_changes(discovered)
                if not needs_rebuild:
                    changes = self.manifest.classify(discovered)
                    logger.info(
                        "Index manifest compared for safe lifecycle",
                        extra={f"manifest_{key}": len(value) for key, value in changes.items()},
                    )
                    needs_rebuild = any(changes[key] for key in ("added", "modified", "deleted"))
                if not needs_rebuild and self._load_vectorstore():
                    self._save_manifest(discovered)
                    return

                logger.info("Building replacement vector store in staging")
                self._replace_index_and_manifest_atomically(splits, discovered)
        except FileLockTimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for the RAG index lock. Another process may be indexing this data directory."
            ) from exc

    def _cleanup_abandoned_index_artifacts(
        self,
        staging_manifest: Path | None = None,
        backup_manifest: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        if staging_manifest is not None and staging_manifest.exists():
            logger.warning("Removing abandoned RAG manifest staging file", extra={"path": str(staging_manifest)})
            staging_manifest.unlink()
        if backup_manifest is not None and manifest_path is not None and backup_manifest.exists() and manifest_path.exists():
            logger.warning("Removing stale RAG manifest backup file", extra={"path": str(backup_manifest)})
            backup_manifest.unlink()
        elif backup_manifest is not None and manifest_path is not None and backup_manifest.exists() and not manifest_path.exists():
            logger.warning("Restoring RAG manifest backup because active manifest is missing", extra={"path": str(backup_manifest)})
            backup_manifest.replace(manifest_path)

    def _validate_manifest_file(self, manifest_path: Path) -> None:
        manifest = IndexManifest(manifest_path).load()
        if manifest.corrupt or not manifest.entries:
            raise RuntimeError(f"RAG manifest validation failed for {manifest_path}.")

    def _write_manifest(self, manifest_path: Path, discovered: dict[str, ManifestEntry]) -> None:
        manifest = IndexManifest(manifest_path)
        manifest.entries = discovered
        manifest.corrupt = False
        manifest.save()

    def _transaction_journal_path(self) -> Path:
        return self.settings.data_dir / "rag_index_transaction.json"

    def _write_transaction_journal(self, state: str, **paths: Path) -> None:
        journal_path = self._transaction_journal_path()
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "transaction_version": TRANSACTION_VERSION,
            "state": state,
            **{key: str(value) for key, value in paths.items()},
        }
        fd, tmp_name = tempfile.mkstemp(prefix=journal_path.name, suffix=".tmp", dir=str(journal_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_name, journal_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _recover_index_transaction(self) -> None:
        journal_path = self._transaction_journal_path()
        if not journal_path.exists():
            return
        try:
            payload = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("RAG index transaction journal is unreadable; refusing inconsistent startup.") from exc
        if payload.get("transaction_version") != TRANSACTION_VERSION:
            raise RuntimeError("Unsupported RAG index transaction journal version.")

        active_manifest = Path(payload["active_manifest"])
        staging_manifest = Path(payload["staging_manifest"])
        backup_manifest = Path(payload["backup_manifest"])

        if active_manifest.exists():
            try:
                self._validate_manifest_file(active_manifest)
            except Exception:
                if not backup_manifest.exists():
                    raise
                active_manifest.unlink(missing_ok=True)
                backup_manifest.replace(active_manifest)
            staging_manifest.unlink(missing_ok=True)
            backup_manifest.unlink(missing_ok=True)
            journal_path.unlink()
            self.manifest.load()
            return

        if staging_manifest.exists():
            self._validate_manifest_file(staging_manifest)
            staging_manifest.replace(active_manifest)
            backup_manifest.unlink(missing_ok=True)
            journal_path.unlink()
            self.manifest.load()
            return

        if backup_manifest.exists():
            backup_manifest.replace(active_manifest)
            journal_path.unlink()
            self.manifest.load()
            return

        journal_path.unlink()
        self.manifest.load()

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
