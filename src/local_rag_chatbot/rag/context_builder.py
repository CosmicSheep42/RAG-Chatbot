from __future__ import annotations

from typing import Any


class ContextBuilder:
    def __init__(self, *, max_context_tokens: int = 6000) -> None:
        self.max_context_tokens = max(1, max_context_tokens)

    def format_context(self, documents: list[Any]) -> str:
        context_blocks: list[str] = []
        used_tokens = 0

        for index, document in enumerate(self._unique_documents(documents), start=1):
            metadata = getattr(document, "metadata", {}) or {}

            source = str(metadata.get("source", "")).strip()
            header_path = str(metadata.get("header_path", "")).strip()

            metadata_lines: list[str] = []

            if source:
                metadata_lines.append(f"Source file: {source}")

            if header_path:
                if metadata.get("is_section_group"):
                    metadata_lines.append(f"Section: {header_path}")
                else:
                    metadata_lines.append(f"Headers: {header_path}")

            block_parts = [f'<DOCUMENT_CONTEXT reference="S{index}">']

            if metadata_lines:
                block_parts.extend(
                    [
                        "<DOCUMENT_METADATA>",
                        "\n".join(metadata_lines),
                        "</DOCUMENT_METADATA>",
                    ]
                )

            block_parts.extend(
                [
                    "<UNTRUSTED_DOCUMENT_CONTENT>",
                    document.page_content.strip(),
                    "</UNTRUSTED_DOCUMENT_CONTENT>",
                    "</DOCUMENT_CONTEXT>",
                ]
            )

            block = "\n".join(block_parts)
            block_tokens = max(1, (len(block) + 3) // 4)
            if context_blocks and used_tokens + block_tokens > self.max_context_tokens:
                break
            context_blocks.append(block)
            used_tokens += block_tokens

        return "\n\n".join(context_blocks)

    def build_citations(self, documents: list[Any]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        unique_documents = self._unique_documents(documents)
        allowed = self.format_context(unique_documents).count('reference="')
        for index, document in enumerate(unique_documents[:allowed], start=1):
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

    def _unique_documents(self, documents: list[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[tuple[Any, Any, Any]] = set()
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            key = (metadata.get("document_id"), metadata.get("chunk_id"), metadata.get("source", "unknown"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(document)
        return unique
