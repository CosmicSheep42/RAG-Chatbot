from __future__ import annotations

import hashlib
import json
from pathlib import Path


class HashManager:
    def __init__(self, hash_file: Path) -> None:
        self.hash_file = hash_file

    def compute_directory_hash(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        ignored_paths: set[Path] | None = None,
        ignored_directory_paths: set[Path] | None = None,
    ) -> str:
        if patterns is None:
            patterns = ["*.txt", "*.md", "*.json"]

        if not directory.exists():
            return ""

        hasher = hashlib.sha256()
        files: list[Path] = []
        resolved_directory = directory.resolve()
        resolved_ignored_paths = {
            self._resolve_configured_path(resolved_directory, path)
            for path in (ignored_paths or set())
        }
        resolved_ignored_directories = {
            resolved_path
            for path in (ignored_directory_paths or set())
            if (resolved_path := self._resolve_configured_path(resolved_directory, path)).is_relative_to(resolved_directory)
        }
        for pattern in patterns:
            files.extend(directory.rglob(pattern))

        for file_path in sorted(files):
            resolved_file_path = file_path.resolve()
            if file_path.is_file() and resolved_file_path not in resolved_ignored_paths:
                if any(
                    resolved_file_path == ignored_dir or resolved_file_path.is_relative_to(ignored_dir)
                    for ignored_dir in resolved_ignored_directories
                ):
                    continue
                rel_path = str(file_path.relative_to(directory.parent))
                hasher.update(rel_path.encode("utf-8"))
                with file_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def _resolve_configured_path(self, directory: Path, path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (directory / path).resolve()

    def load_stored_hash(self, key: str) -> str:
        if not self.hash_file.exists():
            return ""

        try:
            with self.hash_file.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
            return stored.get(key, "")
        except (json.JSONDecodeError, OSError):
            return ""

    def save_hash(self, key: str, hash_value: str) -> None:
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)

        stored: dict[str, str] = {}
        if self.hash_file.exists():
            try:
                with self.hash_file.open("r", encoding="utf-8") as handle:
                    stored = json.load(handle)
            except (json.JSONDecodeError, OSError):
                stored = {}

        stored[key] = hash_value
        with self.hash_file.open("w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2)
