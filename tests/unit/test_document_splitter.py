from langchain_core.documents import Document

from local_rag_chatbot.documents.document_splitter import DocumentSplitter


def test_markdown_split_preserves_heading_metadata_and_source():
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=5)

    chunks = splitter.split(
        [Document(page_content="# Billing\nInvoices are due within thirty days.", metadata={"source": "guide.md"})]
    )

    assert len(chunks) == 1
    assert chunks[0].page_content.startswith("# Billing")
    assert chunks[0].metadata["source"] == "guide.md"
    assert chunks[0].metadata["headings"] == ["Billing"]
    assert chunks[0].metadata["hierarchy_path"] == "Billing"
    assert chunks[0].metadata["chunk_index"] == 0


def test_empty_document_produces_no_chunks():
    assert DocumentSplitter(chunk_size=20, chunk_overlap=0).split([Document(page_content="")]) == []


def test_repeated_split_has_stable_chunk_identity_except_timestamp():
    document = Document(page_content="A short document", metadata={"source": "a.txt"})
    splitter = DocumentSplitter(chunk_size=50, chunk_overlap=0)

    first = splitter.split([document])[0].metadata
    second = splitter.split([document])[0].metadata

    assert first["chunk_id"] == second["chunk_id"]
    assert first["document_id"] == second["document_id"]


def test_source_offsets_use_verified_unicode_source_coordinates():
    source = "# Café\n\nAlpha\n\nAlpha\n"
    chunks = DocumentSplitter(chunk_size=10, chunk_overlap=0).split(
        [Document(page_content=source, metadata={"source": "guide.md", "_source_text": source})]
    )

    assert [chunk.metadata["source_char_start"] for chunk in chunks] == [0, 8, 15]
    for chunk in chunks:
        start = chunk.metadata["source_char_start"]
        end = chunk.metadata["source_char_end"]
        assert source[start:end].strip() == chunk.page_content.strip()


def test_unverifiable_source_offsets_are_not_fabricated():
    chunks = DocumentSplitter(chunk_size=50, chunk_overlap=0).split(
        [Document(page_content="normalized text", metadata={"source": "guide.md", "_source_text": "other"})]
    )
    assert "source_char_start" not in chunks[0].metadata


def test_source_offsets_accept_verified_whitespace_normalization():
    chunks = DocumentSplitter(chunk_size=50, chunk_overlap=0).split(
        [Document(page_content="Alpha\nBeta", metadata={"source": "guide.md", "_source_text": "Alpha  \n\nBeta"})]
    )
    assert chunks[0].metadata["source_char_start"] == 0
    assert chunks[0].metadata["source_char_end"] == len("Alpha  \n\nBeta")
