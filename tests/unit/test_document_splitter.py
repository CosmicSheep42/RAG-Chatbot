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
