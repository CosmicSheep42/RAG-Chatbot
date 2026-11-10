from __future__ import annotations

from typing import Any


class ContextBuilder:
    def __init__(self, *, max_context_tokens: int = 6000, reserved_prompt_tokens: int = 2048) -> None:
        self.max_context_tokens = max(1, max_context_tokens)
        self.reserved_prompt_tokens = max(0, reserved_prompt_tokens)

    def format_context(self, documents: list[Any]) -> str:
        return "\n\n".join(self._format_document(document, index) for index, document in enumerate(self._select_documents(documents), start=1))

    def build_citations(self, documents: list[Any]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for index, document in enumerate(self._select_documents(documents), start=1):
            metadata = getattr(document, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            document_id = metadata.get("document_id")
            source = metadata.get("source", "unknown")
            citation = {
                "citation_id": f"S{index}",
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source": source,
                "source_name": metadata.get("source_name") or source,
                "file_type": metadata.get("file_type"),
                "page_number": metadata.get("page_number"),
                "slide_number": metadata.get("slide_number"),
                "sheet_name": metadata.get("sheet_name"),
                "section": metadata.get("section") or metadata.get("hierarchy_path") or metadata.get("header_path"),
                "snippet": (document.page_content[:300] if hasattr(document, "page_content") else str(document)[:300]),
            }
            citations.append({key: value for key, value in citation.items() if value is not None})
        return citations

    def _select_documents(self, documents: list[Any]) -> list[Any]:
        available_tokens = max(0, self.max_context_tokens - self.reserved_prompt_tokens)
        selected: list[Any] = []
        used_tokens = 0
        for document in self._unique_documents(documents):
            block = self._format_document(document, len(selected) + 1)
            block_tokens = self._estimate_tokens(block)
            if block_tokens > available_tokens or used_tokens + block_tokens > available_tokens:
                continue
            selected.append(document)
            used_tokens += block_tokens
        return selected

    def _format_document(self, document: Any, index: int) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        source = str(metadata.get("source", "")).strip()
        header_path = str(metadata.get("header_path", "")).strip()
        metadata_lines: list[str] = []
        if source:
            metadata_lines.append(f"Source file: {source}")
        if header_path:
            label = "Section" if metadata.get("is_section_group") else "Headers"
            metadata_lines.append(f"{label}: {header_path}")
        block_parts = [f'<DOCUMENT_CONTEXT reference="S{index}">']
        if metadata_lines:
            block_parts.extend(["<DOCUMENT_METADATA>", "\n".join(metadata_lines), "</DOCUMENT_METADATA>"])
        block_parts.extend(
            [
                "<UNTRUSTED_DOCUMENT_CONTENT>",
                str(getattr(document, "page_content", "")).strip(),
                "</UNTRUSTED_DOCUMENT_CONTENT>",
                "</DOCUMENT_CONTEXT>",
            ]
        )
        return "\n".join(block_parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Conservative approximation for providers without a shared tokenizer.
        return max(1, (len(text) + 3) // 4)

    def _unique_documents(self, documents: list[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[tuple[Any, ...]] = set()
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            source = metadata.get("source", "unknown")
            key = (
                metadata.get("document_id"),
                metadata.get("chunk_id"),
                source,
                (
                    str(getattr(document, "page_content", ""))
                    if not metadata.get("document_id") and not metadata.get("chunk_id")
                    else ""
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        return unique
