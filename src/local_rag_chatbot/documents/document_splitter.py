from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
HTML_HEADING_PATTERN = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
JSON_PATH_PATTERN = re.compile(r"^\s*(\$[.\[][^\s:]*):")
TABLE_ROW_PATTERN = re.compile(r"^Row\s+\d+:\s+", re.IGNORECASE)


@dataclass(frozen=True)
class StructuralUnit:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: Iterable[Document]) -> list[Document]:
        split_documents: list[Document] = []
        indexed_at = datetime.now(timezone.utc).isoformat()
        source_cursors: dict[str, int] = {}

        for document in documents:
            base_metadata = dict(document.metadata)
            source_text = base_metadata.pop("_source_text", None)
            units = self._structural_units(document)
            document_id = str(base_metadata.get("document_id") or self._stable_hash(base_metadata.get("source", "")))
            document_chunk_index = 0

            for unit in units:
                unit_metadata = dict(base_metadata)
                unit_metadata.update(unit.metadata)
                for chunk_content, split_index in self._split_unit(unit.content, unit_metadata):
                    content = chunk_content.strip()
                    if not content:
                        continue

                    chunk_index = document_chunk_index
                    document_chunk_index += 1
                    metadata = dict(unit_metadata)
                    metadata["document_id"] = document_id
                    metadata["chunk_index"] = chunk_index
                    metadata["chunk_id"] = self._chunk_id(document_id, chunk_index, metadata, content)
                    metadata["chunk_size"] = len(content)
                    metadata["chunk_size_limit"] = self.chunk_size
                    metadata["structural_split_index"] = split_index
                    metadata["indexed_at"] = indexed_at
                    metadata["index_version"] = "structural-v2"
                    if isinstance(source_text, str):
                        source_key = str(metadata.get("source", ""))
                        span = self._verified_source_span(source_text, content, source_cursors.get(source_key, 0))
                        if span is not None:
                            metadata["source_char_start"], metadata["source_char_end"] = span
                            source_cursors[source_key] = span[0] + 1
                    metadata.setdefault("source_path", metadata.get("source"))
                    metadata.setdefault("header_path", str(metadata.get("hierarchy_path") or ""))
                    if len(content) > self.chunk_size:
                        metadata["chunk_size_limit_exceeded"] = True
                        metadata["chunk_size_limit_exceeded_reason"] = "atomic_unit_exceeds_chunk_size"

                    split_documents.append(Document(page_content=content, metadata=metadata))

        return split_documents

    def _verified_source_span(self, source: str, content: str, search_from: int) -> tuple[int, int] | None:
        """Return a source span only when the chunk can be verified in the original text."""
        start = source.find(content, max(search_from, 0))
        if start >= 0 and source[start : start + len(content)] == content:
            return start, start + len(content)

        normalized_source, positions = self._normalized_with_positions(source)
        normalized_content = re.sub(r"\s+", " ", content).strip()
        if not normalized_content:
            return None
        normalized_start = normalized_source.find(normalized_content)
        while normalized_start >= 0 and positions[normalized_start] < search_from:
            normalized_start = normalized_source.find(normalized_content, normalized_start + 1)
        if normalized_start < 0:
            return None
        start = positions[normalized_start]
        end = positions[normalized_start + len(normalized_content) - 1] + 1
        if re.sub(r"\s+", " ", source[start:end]).strip() == normalized_content:
            return start, end
        return None

    def _normalized_with_positions(self, value: str) -> tuple[str, list[int]]:
        normalized: list[str] = []
        positions: list[int] = []
        pending_space = False
        for index, character in enumerate(value):
            if character.isspace():
                pending_space = bool(normalized)
                continue
            if pending_space:
                normalized.append(" ")
                positions.append(index - 1)
                pending_space = False
            normalized.append(character)
            positions.append(index)
        return "".join(normalized), positions

    def _structural_units(self, document: Document) -> list[StructuralUnit]:
        file_type = str(document.metadata.get("file_type") or "").lower()
        source = str(document.metadata.get("source") or "").lower()
        explicit_structural_type = document.metadata.get("structural_type")

        if explicit_structural_type:
            unit_metadata = {
                "structural_type": explicit_structural_type,
                "hierarchy_path": str(document.metadata.get("hierarchy_path") or ""),
            }
            for key in (
                "headings",
                "heading_level",
                "section_index",
                "table_index",
                "page_number",
                "slide_number",
                "sheet_name",
                "json_path",
            ):
                if key in document.metadata:
                    unit_metadata[key] = document.metadata[key]
            return [StructuralUnit(document.page_content, unit_metadata)]

        if file_type in {"md", "markdown"} or source.endswith(".md"):
            return self._markdown_units(document.page_content)
        if file_type in {"html", "htm"} or source.endswith((".html", ".htm")):
            return self._html_units(document.page_content)
        if file_type in {"csv", "xlsx"} or "sheet_name" in document.metadata:
            return self._table_units(document.page_content, document.metadata)
        if file_type == "json" or source.endswith(".json"):
            return self._json_units(document.page_content)

        metadata: dict[str, Any] = {}
        if "page_number" in document.metadata:
            metadata["structural_type"] = "page"
            metadata["hierarchy_path"] = f"page:{document.metadata['page_number']}"
        elif "slide_number" in document.metadata:
            metadata["structural_type"] = "slide"
            metadata["hierarchy_path"] = f"slide:{document.metadata['slide_number']}"
        elif document.metadata.get("headings"):
            metadata["structural_type"] = "section"
            metadata["hierarchy_path"] = self._join_path(document.metadata.get("headings"))
        else:
            metadata["structural_type"] = "text"
        return [StructuralUnit(document.page_content, metadata)]

    def _markdown_units(self, content: str) -> list[StructuralUnit]:
        units: list[StructuralUnit] = []
        current_lines: list[str] = []
        heading_stack: list[str] = []

        for line in content.splitlines():
            match = MARKDOWN_HEADING_PATTERN.match(line)
            if match and current_lines:
                units.append(self._markdown_unit("\n".join(current_lines), heading_stack))
                current_lines = []

            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)

            current_lines.append(line)

        if current_lines:
            units.append(self._markdown_unit("\n".join(current_lines), heading_stack))

        return units or [StructuralUnit(content, {"structural_type": "markdown"})]

    def _markdown_unit(self, content: str, heading_stack: list[str]) -> StructuralUnit:
        return StructuralUnit(
            content,
            {
                "structural_type": "section",
                "headings": list(heading_stack),
                "hierarchy_path": self._join_path(heading_stack),
            },
        )

    def _html_units(self, content: str) -> list[StructuralUnit]:
        matches = list(HTML_HEADING_PATTERN.finditer(content))
        if not matches:
            return [StructuralUnit(content, {"structural_type": "html"})]

        units: list[StructuralUnit] = []
        heading_stack: list[str] = []
        for index, match in enumerate(matches):
            level = int(match.group(1))
            title = self._strip_tags(match.group(2)).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            section = content[match.start() : end].strip()
            units.append(
                StructuralUnit(
                    section,
                    {
                        "structural_type": "section",
                        "headings": list(heading_stack),
                        "hierarchy_path": self._join_path(heading_stack),
                    },
                )
            )
        return units

    def _table_units(self, content: str, metadata: dict[str, Any]) -> list[StructuralUnit]:
        lines = [line for line in content.splitlines() if line.strip()]
        header_lines = [line for line in lines if not TABLE_ROW_PATTERN.match(line)]
        row_lines = [line for line in lines if TABLE_ROW_PATTERN.match(line)]
        if not row_lines:
            return [StructuralUnit(content, {"structural_type": "table", "hierarchy_path": self._table_path(metadata)})]

        units: list[StructuralUnit] = []
        current_rows: list[str] = []
        for row in row_lines:
            candidate = "\n".join(header_lines + current_rows + [row])
            if current_rows and len(candidate) > self.chunk_size:
                units.append(self._table_unit(header_lines, current_rows, metadata))
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            units.append(self._table_unit(header_lines, current_rows, metadata))
        return units

    def _table_unit(self, header_lines: list[str], rows: list[str], metadata: dict[str, Any]) -> StructuralUnit:
        return StructuralUnit(
            "\n".join(header_lines + rows),
            {
                "structural_type": "table",
                "hierarchy_path": self._table_path(metadata),
                "table_row_count": len(rows),
            },
        )

    def _json_units(self, content: str) -> list[StructuralUnit]:
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        for line in content.splitlines():
            match = JSON_PATH_PATTERN.match(line)
            key = "$"
            if match:
                key = self._json_group_key(match.group(1))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(line)

        return [
            StructuralUnit(
                "\n".join(groups[key]),
                {"structural_type": "json", "hierarchy_path": key, "json_path": key},
            )
            for key in order
        ] or [StructuralUnit(content, {"structural_type": "json", "hierarchy_path": "$"})]

    def _split_unit(self, content: str, metadata: dict[str, Any]) -> Iterable[tuple[str, int]]:
        if len(content) <= self.chunk_size:
            yield content, 0
            return

        if metadata.get("structural_type") == "table":
            yield from self._split_large_table_unit(content)
            return

        document = Document(page_content=content, metadata=dict(metadata))
        for index, chunk in enumerate(self.recursive_splitter.split_documents([document])):
            yield chunk.page_content, index

    def _split_large_table_unit(self, content: str) -> Iterable[tuple[str, int]]:
        lines = [line for line in content.splitlines() if line.strip()]
        header_lines = [line for line in lines if not TABLE_ROW_PATTERN.match(line)]
        row_lines = [line for line in lines if TABLE_ROW_PATTERN.match(line)]
        if not row_lines:
            document = Document(page_content=content, metadata={})
            for index, chunk in enumerate(self.recursive_splitter.split_documents([document])):
                yield chunk.page_content, index
            return

        current_rows: list[str] = []
        split_index = 0
        for row in row_lines:
            candidate = "\n".join(header_lines + current_rows + [row])
            if current_rows and len(candidate) > self.chunk_size:
                yield "\n".join(header_lines + current_rows), split_index
                split_index += 1
                current_rows = self._overlap_rows(current_rows)
            current_rows.append(row)
        if current_rows:
            yield "\n".join(header_lines + current_rows), split_index

    def _overlap_rows(self, rows: list[str]) -> list[str]:
        if self.chunk_overlap <= 0:
            return []
        overlap: list[str] = []
        total = 0
        for row in reversed(rows):
            if overlap and total + len(row) > self.chunk_overlap:
                break
            overlap.insert(0, row)
            total += len(row)
        return overlap

    def _json_group_key(self, path: str) -> str:
        parts = re.split(r"(?=\.)|(?=\[)", path)
        if len(parts) <= 2:
            return path
        return "".join(parts[:2])

    def _table_path(self, metadata: dict[str, Any]) -> str:
        if metadata.get("sheet_name"):
            return f"sheet:{metadata['sheet_name']}"
        return "table"

    def _normalize_header_metadata(self, chunk: Document) -> None:
        headings = chunk.metadata.get("headings")
        if isinstance(headings, list):
            chunk.metadata["header_path"] = self._join_path(headings)
        else:
            chunk.metadata["header_path"] = str(chunk.metadata.get("hierarchy_path") or "")

    def _join_path(self, parts: Any) -> str:
        if isinstance(parts, (list, tuple)):
            return " > ".join(str(part).strip() for part in parts if str(part).strip())
        return str(parts or "")

    def _strip_tags(self, value: str) -> str:
        return re.sub(r"<[^>]+>", "", value)

    def _chunk_id(self, document_id: str, chunk_index: int, metadata: dict[str, Any], content: str) -> str:
        stable_parts = [
            document_id,
            str(chunk_index),
            str(metadata.get("source", "")),
            str(metadata.get("hierarchy_path", "")),
            str(metadata.get("page_number", "")),
            str(metadata.get("slide_number", "")),
            str(metadata.get("sheet_name", "")),
            self._stable_hash(content),
        ]
        return self._stable_hash("|".join(stable_parts))

    def _stable_hash(self, value: Any) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
