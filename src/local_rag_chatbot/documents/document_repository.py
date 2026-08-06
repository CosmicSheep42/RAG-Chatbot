from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from local_rag_chatbot.documents.loaders import DocumentLoaderRegistry
from local_rag_chatbot.documents.ingestion_security import DocumentSecurityValidator, IngestionSecurityConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRepositoryError:
    source: str
    reason: str
    message: str


class DocumentRepository:
    def __init__(
        self,
        documents_dir: Path,
        *,
        loader_registry: DocumentLoaderRegistry | None = None,
        max_file_size_bytes: int = 25 * 1024 * 1024,
        ignored_paths: set[Path] | None = None,
        ignored_directory_paths: set[Path] | None = None,
        security_validator: DocumentSecurityValidator | None = None,
    ) -> None:
        self.documents_dir = documents_dir.resolve()
        self.loader_registry = loader_registry or DocumentLoaderRegistry()
        self.max_file_size_bytes = max_file_size_bytes
        self.ignored_paths = {
            self._resolve_configured_path(path)
            for path in (ignored_paths or set())
        }
        self.ignored_directory_paths = {
            resolved_path
            for path in (ignored_directory_paths or set())
            if (resolved_path := self._resolve_configured_path(path)).is_relative_to(self.documents_dir)
        }
        self.errors: list[DocumentRepositoryError] = []
        self._hash_cache: dict[Path, str] = {}
        self.security_validator = security_validator or DocumentSecurityValidator(
            self.documents_dir,
            IngestionSecurityConfig(max_file_size_bytes=max_file_size_bytes),
        )

    def load(self) -> list[Document]:
        self.errors = []
        self._hash_cache = {}
        if not self.documents_dir.exists():
            logger.info("Documents directory not found")
            return []

        logger.info("Loading documents", extra={"documents_dir": str(self.documents_dir)})

        documents: list[Document] = []
        for path in sorted(self.documents_dir.rglob("*")):
            if not path.is_file() or self._is_ignored(path):
                continue

            loader = self.loader_registry.get_loader(path)
            if loader is None:
                self._record_error(path, "unsupported_format", "No loader registered for this file.")
                continue

            valid, reason = self.security_validator.validate(path)
            if not valid:
                self._record_error(path, reason, "Document failed ingestion security validation.")
                continue

            try:
                file_size = path.stat().st_size
            except OSError as exc:
                self._record_error(path, "stat_failed", str(exc))
                continue

            if file_size > self.max_file_size_bytes:
                self._record_error(
                    path,
                    "file_too_large",
                    f"File size {file_size} exceeds limit {self.max_file_size_bytes}.",
                )
                continue

            base_metadata = self._base_metadata(path, file_size)
            try:
                for loaded_document in loader.load(path, base_metadata=base_metadata):
                    content = loaded_document.page_content.strip()
                    if not content:
                        continue
                    metadata = self._normalize_metadata(path, loaded_document.metadata)
                    documents.append(Document(page_content=content, metadata=metadata))
            except Exception as exc:
                self._record_error(path, "load_failed", str(exc))

        logger.info("Loaded documents", extra={"document_count": len(documents)})
        if self.errors:
            logger.warning("Skipped documents", extra={"skipped_count": len(self.errors)})
        return documents

    def _is_ignored(self, path: Path) -> bool:
        resolved_path = path.resolve()
        if resolved_path in self.ignored_paths:
            return True
        return any(
            resolved_path == ignored_dir or resolved_path.is_relative_to(ignored_dir)
            for ignored_dir in self.ignored_directory_paths
        )

    def _resolve_configured_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (self.documents_dir / path).resolve()

    def _base_metadata(self, path: Path, file_size: int) -> dict[str, Any]:
        relative_path = path.relative_to(self.documents_dir)
        source = str(relative_path)
        return {
            "source": source,
            "source_path": source,
            "source_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "document_id": self._document_id(relative_path),
            "document_hash": self._cached_file_hash(path),
            "file_size_bytes": file_size,
        }

    def _normalize_metadata(self, path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metadata)
        normalized.setdefault("source", str(path.relative_to(self.documents_dir)))
        normalized.setdefault("source_path", normalized["source"])
        normalized.setdefault("source_name", path.name)
        normalized.setdefault("file_type", path.suffix.lower().lstrip("."))
        normalized.setdefault("document_id", self._document_id(path.relative_to(self.documents_dir)))
        if "document_hash" not in normalized:
            normalized["document_hash"] = self._cached_file_hash(path)
        return normalized

    def _record_error(self, path: Path, reason: str, message: str) -> None:
        try:
            source = str(path.resolve().relative_to(self.documents_dir))
        except (OSError, ValueError):
            source = path.name
        error = DocumentRepositoryError(
            source=source,
            reason=reason,
            message=message,
        )
        self.errors.append(error)
        logger.warning("Skipping document %s (%s): %s", error.source, reason, message)

    def _document_id(self, relative_path: Path) -> str:
        return hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:16]

    def _file_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _cached_file_hash(self, path: Path) -> str:
        resolved_path = path.resolve()
        if resolved_path not in self._hash_cache:
            self._hash_cache[resolved_path] = self._file_hash(resolved_path)
        return self._hash_cache[resolved_path]
