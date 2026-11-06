import json
from pathlib import Path

import pytest

from evaluation.validate_dataset import validate_dataset
from local_rag_chatbot.documents.document_repository import DocumentRepository
from local_rag_chatbot.documents.document_splitter import DocumentSplitter


EVALUATION_ROOT = Path(__file__).parents[2] / "evaluation"
DATASET = EVALUATION_ROOT / "dataset" / "questions.jsonl"
CORPUS = EVALUATION_ROOT / "corpus"


def _write_dataset(tmp_path: Path, mutate) -> Path:
    entries = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    mutate(entries)
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return path


def test_committed_dataset_is_valid():
    assert validate_dataset(CORPUS, DATASET) == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda entries: entries.append(dict(entries[0])), "duplicate id"),
        (lambda entries: entries[0]["relevant_documents"].append("missing.md"), "missing source document"),
        (lambda entries: entries[0]["evidence"][0].update(text="not in the source"), "evidence text"),
        (lambda entries: entries[0].update(answerable=False), "null reference_answer"),
        (lambda entries: entries[0].update(split="staging"), "invalid split"),
    ],
)
def test_invalid_dataset_is_rejected(tmp_path: Path, mutation, expected: str):
    errors = validate_dataset(CORPUS, _write_dataset(tmp_path, mutation))
    assert any(expected in error for error in errors)


def test_corpus_loads_with_source_metadata_and_splits(tmp_path: Path):
    documents_dir = tmp_path / "corpus"
    documents_dir.mkdir()
    for source in CORPUS.glob("*.md"):
        (documents_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    documents = DocumentRepository(documents_dir).load()
    chunks = DocumentSplitter(chunk_size=500, chunk_overlap=50).split(documents)

    assert len(documents) == 6
    assert chunks
    assert {document.metadata["source"] for document in documents} == {
        "configuration.md",
        "deployment.md",
        "errors.md",
        "operations.md",
        "platform-overview.md",
        "versions.md",
    }
    assert all(document.metadata["source"] for document in chunks)
    assert any("NBL-429" in document.page_content for document in chunks)
