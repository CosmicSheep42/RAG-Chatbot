import json
from pathlib import Path

import pytest

from evaluation.benchmark import (
    BM25Retriever,
    build_retriever,
    load_corpus,
    load_examples,
    run_end_to_end,
    run_retrieval,
)
from local_rag_chatbot.vectorstores.contracts import ScoredDocument


def test_bm25_benchmark_smoke_produces_structured_results(tmp_path):
    corpus = load_corpus(Path("evaluation/corpus"))
    examples = load_examples(Path("evaluation/dataset/questions.jsonl"), "dev")[:2]

    retrieval = run_retrieval(examples, BM25Retriever(corpus), configuration="bm25", split="dev", top_k=3)
    assert retrieval["records"]
    assert retrieval["records"][0]["status"] == "success"
    json.dumps(retrieval, allow_nan=False)

    class FakeAnswerer:
        def answer(self, question, documents):
            return (documents[0].page_content if documents else "I do not have enough context."), [
                documents[0].metadata["source"]
            ] if documents else []

    end_to_end = run_end_to_end(
        examples[:1],
        BM25Retriever(corpus),
        FakeAnswerer(),
        configuration="bm25",
        split="dev",
        top_k=3,
    )
    assert end_to_end["records"][0]["answer"]
    assert "timing" in end_to_end["records"][0]


def test_existing_vector_store_contract_selects_dense_and_hybrid_adapters(document):
    documents = [document("billing policy", source="guide.md")]

    class Backend:
        def search(self, *_args, **_kwargs):
            return [ScoredDocument(documents[0], 1.0)]

    backend = Backend()
    assert build_retriever("dense", documents, vector_backend=backend).retrieve("billing", 1)
    assert build_retriever("hybrid", documents, vector_backend=backend).retrieve("billing", 1)


def test_benchmark_does_not_mislabel_unavailable_reranking_or_full_pipeline(document):
    documents = [document("billing policy", source="guide.md")]

    class Backend:
        def search(self, *_args, **_kwargs):
            return [ScoredDocument(documents[0], 1.0)]

    with pytest.raises(ValueError, match="hybrid-rerank is unavailable"):
        build_retriever("hybrid-rerank", documents, vector_backend=Backend())
    with pytest.raises(ValueError, match="production ChatbotEngine pipeline"):
        build_retriever("full", documents, vector_backend=Backend())
