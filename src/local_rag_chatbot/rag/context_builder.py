from __future__ import annotations

from typing import Any


class ContextBuilder:
    def format_context(self, documents: list[Any]) -> str:
        context_blocks: list[str] = []

        for document in documents:
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

            block_parts = ["<DOCUMENT_CONTEXT>"]

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

            context_blocks.append("\n".join(block_parts))

        return "\n\n".join(context_blocks)

    def build_citations(self, documents: list[Any]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen = set()
        for index, document in enumerate(documents, start=1):
            metadata = getattr(document, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            document_id = metadata.get("document_id")
            source = metadata.get("source", "unknown")
            dedupe_key = (document_id, chunk_id, source)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
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
