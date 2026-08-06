from __future__ import annotations

import csv
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


class DocumentLoadError(Exception):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    supported_extensions: tuple[str, ...]
    supported_mime_types: tuple[str, ...]

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        ...


class OptionalOcrProvider(Protocol):
    def extract_text(self, path: Path, *, page_number: int) -> str:
        ...


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(f"Could not decode text file: {path}")


def _metadata(base_metadata: dict[str, Any], **values: Any) -> dict[str, Any]:
    metadata = dict(base_metadata)
    metadata.update({key: value for key, value in values.items() if value not in (None, "")})
    return metadata


class PlainTextLoader:
    supported_extensions = (".txt",)
    supported_mime_types = ("text/plain",)

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        yield LoadedDocument(_read_text(path), _metadata(base_metadata))


class MarkdownLoader:
    supported_extensions = (".md",)
    supported_mime_types = ("text/markdown", "text/x-markdown")

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        yield LoadedDocument(_read_text(path), _metadata(base_metadata))


class CsvLoader:
    supported_extensions = (".csv",)
    supported_mime_types = ("text/csv", "application/csv")

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(handle, dialect)
            rows = list(reader)

        if not rows:
            return

        headers = [cell.strip() or f"Column {index + 1}" for index, cell in enumerate(rows[0])]
        lines = [f"Columns: {', '.join(headers)}"]
        for row_index, row in enumerate(rows[1:], start=1):
            values = []
            for column_index, header in enumerate(headers):
                value = row[column_index].strip() if column_index < len(row) else ""
                values.append(f"{header}: {value}")
            lines.append(f"Row {row_index}: " + "; ".join(values))

        yield LoadedDocument(
            "\n".join(lines),
            _metadata(base_metadata, headings=headers),
        )


class JsonLoader:
    supported_extensions = (".json",)
    supported_mime_types = ("application/json",)

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        lines = list(self._flatten(data))
        yield LoadedDocument(
            "\n".join(lines),
            _metadata(base_metadata, sections=self._top_level_sections(data)),
        )

    def _flatten(self, value: Any, prefix: str = "$") -> Iterable[str]:
        if isinstance(value, dict):
            if not value:
                yield f"{prefix}: {{}}"
            for key, child in value.items():
                yield from self._flatten(child, f"{prefix}.{key}")
            return

        if isinstance(value, list):
            if not value:
                yield f"{prefix}: []"
            for index, child in enumerate(value):
                yield from self._flatten(child, f"{prefix}[{index}]")
            return

        yield f"{prefix}: {value}"

    def _top_level_sections(self, data: Any) -> list[str]:
        if isinstance(data, dict):
            return [str(key) for key in data.keys()]
        return []


class HtmlLoader:
    supported_extensions = (".html", ".htm")
    supported_mime_types = ("text/html", "application/xhtml+xml")

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise DocumentLoadError("HTML loading requires beautifulsoup4.") from exc

        soup = BeautifulSoup(_read_text(path), "html.parser")
        for element in soup(["script", "style", "iframe", "object", "embed", "noscript"]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else None
        heading_stack: list[tuple[int, str]] = []
        section_lines: list[str] = []
        section_index = 0

        def flush() -> LoadedDocument | None:
            nonlocal section_lines, section_index
            content = "\n".join(line for line in section_lines if line.strip()).strip()
            if not content:
                section_lines = []
                return None
            headings = [heading for _level, heading in heading_stack]
            metadata = _metadata(
                base_metadata,
                title=title,
                headings=headings,
                hierarchy_path=" > ".join(headings),
                section_index=section_index,
                structural_type="section" if headings else "html",
            )
            section_index += 1
            section_lines = []
            return LoadedDocument(content, metadata)

        for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"]):
            name = node.name.lower()
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            if name.startswith("h"):
                flushed = flush()
                if flushed is not None:
                    yield flushed
                level = int(name[1])
                heading_stack = [(stored_level, heading) for stored_level, heading in heading_stack if stored_level < level]
                heading_stack.append((level, text))
                section_lines = [text]
            elif name == "tr":
                cells = [cell.get_text(" ", strip=True) for cell in node.find_all(["th", "td"])]
                if cells:
                    section_lines.append(" | ".join(cells))
            else:
                section_lines.append(text)

        flushed = flush()
        if flushed is not None:
            yield flushed


class PdfLoader:
    supported_extensions = (".pdf",)
    supported_mime_types = ("application/pdf",)

    def __init__(self, ocr_provider: OptionalOcrProvider | None = None, *, max_pages: int = 1000) -> None:
        self.ocr_provider = ocr_provider
        self.max_pages = max_pages

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentLoadError("PDF loading requires pypdf.") from exc

        reader = PdfReader(str(path))
        for page_index, page in enumerate(reader.pages, start=1):
            if page_index > self.max_pages:
                break
            text = page.extract_text() or ""
            if not text.strip() and self.ocr_provider is not None:
                text = self.ocr_provider.extract_text(path, page_number=page_index)
            if text.strip():
                yield LoadedDocument(
                    text,
                    _metadata(base_metadata, page_number=page_index),
                )


class DocxLoader:
    supported_extensions = (".docx",)
    supported_mime_types = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        try:
            from docx import Document as DocxDocument
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentLoadError("DOCX loading requires python-docx.") from exc

        document = DocxDocument(str(path))
        heading_stack: list[str] = []
        section_lines: list[str] = []
        section_headings: list[str] = []
        section_level: int | None = None
        section_index = 0
        table_index = 0

        def flush_section() -> LoadedDocument | None:
            nonlocal section_lines, section_headings, section_level, section_index
            content = "\n".join(line for line in section_lines if line.strip()).strip()
            if not content:
                section_lines = []
                section_headings = list(heading_stack)
                section_level = len(heading_stack) or None
                return None

            metadata = self._section_metadata(
                base_metadata,
                section_headings,
                section_level,
                section_index,
                structural_type="section" if section_headings else "text",
            )
            section_index += 1
            section_lines = []
            section_headings = list(heading_stack)
            section_level = len(heading_stack) or None
            return LoadedDocument(content, metadata)

        for block in self._iter_block_items(document, Paragraph, Table):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue

                heading_level = self._heading_level(block)
                if heading_level is not None:
                    flushed = flush_section()
                    if flushed is not None:
                        yield flushed

                    heading_stack = heading_stack[: heading_level - 1]
                    heading_stack.append(text)
                    section_headings = list(heading_stack)
                    section_level = heading_level
                    section_lines = [text]
                    continue

                if not section_lines:
                    section_headings = list(heading_stack)
                    section_level = len(heading_stack) or None
                section_lines.append(text)
                continue

            if isinstance(block, Table):
                flushed = flush_section()
                if flushed is not None:
                    yield flushed

                table_index += 1
                table_content = self._format_table(block, table_index)
                if table_content:
                    yield LoadedDocument(
                        table_content,
                        self._section_metadata(
                            base_metadata,
                            heading_stack,
                            len(heading_stack) or None,
                            section_index,
                            structural_type="table",
                            table_index=table_index,
                        ),
                    )
                    section_index += 1

                section_headings = list(heading_stack)
                section_level = len(heading_stack) or None

        flushed = flush_section()
        if flushed is not None:
            yield flushed

    def _iter_block_items(self, document: Any, paragraph_type: type, table_type: type) -> Iterable[Any]:
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                yield paragraph_type(child, document)
            elif child.tag.endswith("}tbl"):
                yield table_type(child, document)

    def _heading_level(self, paragraph: Any) -> int | None:
        style_name = paragraph.style.name if paragraph.style is not None else ""
        match = re.match(r"heading\s+([1-6])$", style_name.strip(), re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))

    def _section_metadata(
        self,
        base_metadata: dict[str, Any],
        headings: list[str],
        heading_level: int | None,
        section_index: int,
        *,
        structural_type: str,
        table_index: int | None = None,
    ) -> dict[str, Any]:
        metadata = _metadata(
            base_metadata,
            headings=list(headings),
            heading_level=heading_level,
            section_index=section_index,
            structural_type=structural_type,
            table_index=table_index,
        )
        metadata["headings"] = list(headings)
        metadata["hierarchy_path"] = " > ".join(headings)
        return metadata

    def _format_table(self, table: Any, table_index: int) -> str:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not rows:
            return ""

        headers = [cell or f"Column {index + 1}" for index, cell in enumerate(rows[0])]
        lines = [f"Table {table_index}:", f"Columns: {', '.join(headers)}"]
        for row_index, row in enumerate(rows[1:], start=1):
            values = []
            for column_index, header in enumerate(headers):
                value = row[column_index] if column_index < len(row) else ""
                values.append(f"{header}: {value}")
            lines.append(f"Row {row_index}: " + "; ".join(values))
        return "\n".join(lines)


class XlsxLoader:
    supported_extensions = (".xlsx",)
    supported_mime_types = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    def __init__(self, *, max_sheets: int = 100, max_rows: int = 100000) -> None:
        self.max_sheets = max_sheets
        self.max_rows = max_rows

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise DocumentLoadError("XLSX loading requires openpyxl.") from exc

        workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
        try:
            for sheet_index, worksheet in enumerate(workbook.worksheets, start=1):
                if sheet_index > self.max_sheets:
                    break
                rows = worksheet.iter_rows(values_only=True)
                try:
                    first_row = next(rows)
                except StopIteration:
                    continue

                headers = [
                    str(value).strip() if value not in (None, "") else f"Column {index + 1}"
                    for index, value in enumerate(first_row)
                ]
                lines = [f"Sheet: {worksheet.title}", f"Columns: {', '.join(headers)}"]
                for row_index, row in enumerate(rows, start=1):
                    if row_index > self.max_rows:
                        break
                    values = []
                    for column_index, header in enumerate(headers):
                        value = row[column_index] if column_index < len(row) else ""
                        values.append(f"{header}: {'' if value is None else value}")
                    lines.append(f"Row {row_index}: " + "; ".join(values))

                yield LoadedDocument(
                    "\n".join(lines),
                    _metadata(base_metadata, sheet_name=worksheet.title, headings=headers),
                )
        finally:
            workbook.close()


class PptxLoader:
    supported_extensions = (".pptx",)
    supported_mime_types = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    def __init__(self, *, max_slides: int = 500) -> None:
        self.max_slides = max_slides

    def load(self, path: Path, *, base_metadata: dict[str, Any]) -> Iterable[LoadedDocument]:
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise DocumentLoadError("PPTX loading requires python-pptx.") from exc

        presentation = Presentation(str(path))
        for slide_index, slide in enumerate(presentation.slides, start=1):
            if slide_index > self.max_slides:
                break
            lines: list[str] = []
            title = None
            if slide.shapes.title and getattr(slide.shapes.title, "text", "").strip():
                title = slide.shapes.title.text.strip()
                lines.append(title)

            for shape in slide.shapes:
                if shape is slide.shapes.title:
                    continue
                if getattr(shape, "has_table", False):
                    lines.extend(self._format_table(shape.table))
                    continue
                if getattr(shape, "has_text_frame", False):
                    text = shape.text_frame.text.strip()
                    if text:
                        lines.append(text)

            content = "\n".join(line for line in lines if line.strip()).strip()
            if content:
                yield LoadedDocument(
                    content,
                    _metadata(
                        base_metadata,
                        slide_number=slide_index,
                        title=title,
                        headings=[title] if title else [],
                        hierarchy_path=title,
                        structural_type="slide",
                    ),
                )

    def _format_table(self, table: Any) -> list[str]:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        return rows


class DocumentLoaderRegistry:
    def __init__(
        self,
        loaders: Iterable[DocumentLoader] | None = None,
        *,
        max_pdf_pages: int = 1000,
        max_spreadsheet_sheets: int = 100,
        max_spreadsheet_rows: int = 100000,
        max_presentation_slides: int = 500,
    ) -> None:
        self._by_extension: dict[str, DocumentLoader] = {}
        self._by_mime_type: dict[str, DocumentLoader] = {}
        default_loaders = loaders or self.default_loaders(
            max_pdf_pages=max_pdf_pages,
            max_spreadsheet_sheets=max_spreadsheet_sheets,
            max_spreadsheet_rows=max_spreadsheet_rows,
            max_presentation_slides=max_presentation_slides,
        )
        for loader in default_loaders:
            self.register(loader)

    @classmethod
    def default_loaders(
        cls,
        *,
        max_pdf_pages: int = 1000,
        max_spreadsheet_sheets: int = 100,
        max_spreadsheet_rows: int = 100000,
        max_presentation_slides: int = 500,
    ) -> tuple[DocumentLoader, ...]:
        return (
            PlainTextLoader(),
            MarkdownLoader(),
            PdfLoader(max_pages=max_pdf_pages),
            DocxLoader(),
            XlsxLoader(max_sheets=max_spreadsheet_sheets, max_rows=max_spreadsheet_rows),
            CsvLoader(),
            JsonLoader(),
            HtmlLoader(),
            PptxLoader(max_slides=max_presentation_slides),
        )

    def register(self, loader: DocumentLoader) -> None:
        for extension in loader.supported_extensions:
            self._by_extension[extension.lower()] = loader
        for mime_type in loader.supported_mime_types:
            self._by_mime_type[mime_type.lower()] = loader

    def get_loader(self, path: Path, mime_type: str | None = None) -> DocumentLoader | None:
        if mime_type:
            loader = self._by_mime_type.get(mime_type.lower())
            if loader is not None:
                return loader

        guessed_mime_type, _encoding = mimetypes.guess_type(path.name)
        if guessed_mime_type:
            loader = self._by_mime_type.get(guessed_mime_type.lower())
            if loader is not None:
                return loader

        return self._by_extension.get(path.suffix.lower())

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_extension.keys()))
