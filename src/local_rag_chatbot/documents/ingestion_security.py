from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile


@dataclass(frozen=True)
class IngestionSecurityConfig:
    max_file_size_bytes: int
    max_zip_entries: int = 1000
    max_zip_uncompressed_bytes: int = 100 * 1024 * 1024
    max_zip_compression_ratio: float = 100.0


class ExternalDocumentScanner:
    def scan(self, path: Path) -> tuple[bool, str | None]:
        return True, None


class DocumentSecurityValidator:
    ZIP_EXTENSIONS = {".docx", ".xlsx", ".pptx"}

    def __init__(
        self,
        root: Path,
        config: IngestionSecurityConfig,
        *,
        scanner: ExternalDocumentScanner | None = None,
    ) -> None:
        self.root = root.resolve()
        self.config = config
        self.scanner = scanner or ExternalDocumentScanner()

    def validate(self, path: Path) -> tuple[bool, str]:
        try:
            resolved = path.resolve()
        except OSError:
            return False, "path_resolve_failed"
        if not resolved.is_relative_to(self.root):
            return False, "path_outside_documents_root"
        if path.is_symlink() and not resolved.is_relative_to(self.root):
            return False, "symlink_outside_documents_root"
        try:
            size = resolved.stat().st_size
        except OSError:
            return False, "stat_failed"
        if size > self.config.max_file_size_bytes:
            return False, "file_too_large"
        ok, reason = self.scanner.scan(resolved)
        if not ok:
            return False, reason or "scanner_rejected"
        if resolved.suffix.lower() in self.ZIP_EXTENSIONS:
            return self._validate_zip_container(resolved, size)
        return True, "ok"

    def _validate_zip_container(self, path: Path, file_size: int) -> tuple[bool, str]:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > self.config.max_zip_entries:
                    return False, "zip_too_many_entries"
                total = sum(max(0, info.file_size) for info in infos)
                compressed = sum(max(0, info.compress_size) for info in infos) or max(1, file_size)
                if total > self.config.max_zip_uncompressed_bytes:
                    return False, "zip_uncompressed_too_large"
                if compressed and total / compressed > self.config.max_zip_compression_ratio:
                    return False, "zip_compression_ratio_too_high"
                suspicious = {".bin", ".exe", ".dll", ".vba", ".js"}
                for info in infos:
                    name = info.filename.lower()
                    if ".." in Path(name).parts or name.startswith("/"):
                        return False, "zip_path_traversal"
                    if "vbaProject.bin".lower() in name or Path(name).suffix in suspicious:
                        return False, "zip_embedded_active_content"
        except zipfile.BadZipFile:
            return False, "zip_malformed"
        except OSError:
            return False, "zip_read_failed"
        return True, "ok"
