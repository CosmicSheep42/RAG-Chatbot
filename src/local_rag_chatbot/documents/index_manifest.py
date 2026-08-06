from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import tempfile
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifestEntry:
    document_id: str
    relative_path: str
    file_hash: str
    file_type: str
    parser_version: str
    chunker_version: str
    embedding_model: str
    embedding_config: str
    indexed_chunk_ids: list[str]
    indexed_at: str


class IndexManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.schema_version = SCHEMA_VERSION
        self.entries: dict[str, ManifestEntry] = {}
        self.corrupt = False

    def load(self) -> "IndexManifest":
        if not self.path.exists():
            return self
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema_version") != SCHEMA_VERSION:
                self.corrupt = True
                return self
            self.entries = {
                item["relative_path"]: ManifestEntry(**item)
                for item in payload.get("documents", [])
            }
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            self.corrupt = True
            self.entries = {}
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "documents": [asdict(entry) for entry in sorted(self.entries.values(), key=lambda item: item.relative_path)],
        }
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def classify(self, discovered: dict[str, ManifestEntry]) -> dict[str, list[str]]:
        current = set(discovered)
        previous = set(self.entries)
        added = sorted(current - previous)
        deleted = sorted(previous - current)
        modified = sorted(
            path
            for path in current & previous
            if discovered[path].file_hash != self.entries[path].file_hash
            or discovered[path].parser_version != self.entries[path].parser_version
            or discovered[path].chunker_version != self.entries[path].chunker_version
            or discovered[path].embedding_model != self.entries[path].embedding_model
            or discovered[path].embedding_config != self.entries[path].embedding_config
        )
        unchanged = sorted((current & previous) - set(modified))
        return {"added": added, "modified": modified, "deleted": deleted, "unchanged": unchanged}

    def has_incompatible_changes(self, discovered: dict[str, ManifestEntry]) -> bool:
        if self.corrupt:
            return True
        for path, entry in discovered.items():
            previous = self.entries.get(path)
            if previous is None:
                continue
            if (
                entry.parser_version != previous.parser_version
                or entry.chunker_version != previous.chunker_version
                or entry.embedding_model != previous.embedding_model
                or entry.embedding_config != previous.embedding_config
            ):
                return True
        return False
