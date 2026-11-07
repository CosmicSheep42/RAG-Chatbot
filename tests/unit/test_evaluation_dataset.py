import json
import shutil
from pathlib import Path

import pytest

from evaluation.adapters.datapizza import load_examples
from evaluation.validate_dataset import validate_dataset
from local_rag_chatbot.documents.document_repository import DocumentRepository
from local_rag_chatbot.documents.document_splitter import DocumentSplitter


ROOT = Path(__file__).parents[2]
DATASET = ROOT / "evaluation" / "dataset" / "datapizza"
CORPUS = ROOT / "knowledge-base"


def _copy_dataset(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "knowledge-base"
    dataset = tmp_path / "datapizza"
    shutil.copytree(CORPUS, corpus)
    shutil.copytree(DATASET, dataset)
    return corpus, dataset


def test_committed_datapizza_dataset_is_valid():
    assert validate_dataset(CORPUS, DATASET) == []
    assert len(load_examples(DATASET, "dev")) == 12
    assert len(load_examples(DATASET, "test")) == 44


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda entries: entries.append(dict(entries[0])), "duplicate normalized id"),
        (lambda entries: entries[0]["passages"][0].update(document_path="missing.md"), "missing source document"),
        (lambda entries: entries[0]["passages"][0].update(content="not in the source"), "content does not match"),
        (lambda entries: entries[0]["passages"][0].update(start_char=-1), "invalid character offsets"),
    ],
)
def test_invalid_datapizza_dataset_is_rejected(tmp_path: Path, mutation, expected: str):
    corpus, dataset = _copy_dataset(tmp_path)
    path = dataset / "easy.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    mutation(entries)
    path.write_text(json.dumps(entries), encoding="utf-8")
    errors = validate_dataset(corpus, dataset)
    assert any(expected in error for error in errors)


def test_corpus_loads_with_source_metadata_and_splits(tmp_path: Path):
    documents_dir = tmp_path / "corpus"
    documents_dir.mkdir()
    for source in CORPUS.glob("*.md"):
        (documents_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    documents = DocumentRepository(documents_dir).load()
    chunks = DocumentSplitter(chunk_size=500, chunk_overlap=50).split(documents)

    assert len(documents) == 20
    assert chunks
    assert {document.metadata["source"] for document in documents} == {path.name for path in CORPUS.glob("*.md")}
    assert all(document.metadata["source"] for document in chunks)
