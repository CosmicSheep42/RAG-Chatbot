import json
from pathlib import Path

import pytest

from evaluation.benchmark import (
    BM25Retriever,
    BenchmarkExample,
    ChatModelAnswerer,
    EngineAnswerer,
    build_retriever,
    load_corpus,
    load_examples,
    run_end_to_end,
    run_retrieval,
)
from local_rag_chatbot.rag.index_manager import RAG_VECTOR_NAMESPACE
from local_rag_chatbot.vectorstores.contracts import ScoredDocument


def test_bm25_benchmark_smoke_produces_structured_results(tmp_path):
    corpus = load_corpus(Path("knowledge-base"))
    examples = load_examples(Path("evaluation/dataset/datapizza"), "dev")[:2]

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
        def index_exists(self, _namespace):
            return True

        def search(self, *_args, **_kwargs):
            return [ScoredDocument(documents[0], 1.0)]

    backend = Backend()
    assert build_retriever("dense", documents, vector_backend=backend).retrieve("billing", 1)
    assert build_retriever("hybrid", documents, vector_backend=backend).retrieve("billing", 1)


def test_dense_retrieval_uses_production_namespace(document):
    documents = [document("billing policy", source="guide.md")]
    calls = []

    class Backend:
        def index_exists(self, _namespace):
            return True

        def search(self, namespace, *_args, **_kwargs):
            calls.append(namespace)
            return [ScoredDocument(documents[0], 1.0)]

    build_retriever("dense", documents, vector_backend=Backend()).retrieve("billing", 1)
    assert calls == [RAG_VECTOR_NAMESPACE]


def test_vector_retrieval_fails_when_production_namespace_is_missing(document):
    class Backend:
        def index_exists(self, _namespace):
            return False

    with pytest.raises(RuntimeError, match="namespace does not exist"):
        build_retriever("dense", [document("billing", source="guide.md")], vector_backend=Backend())


def test_hybrid_reranker_model_is_propagated(monkeypatch, document):
    documents = [document("billing policy", source="guide.md")]
    captured = {}

    class FakeHybridRetriever:
        reranking_active = True
        reranking_status = {"status": "active", "reason": None}

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def retrieve(self, *_args, **_kwargs):
            return documents

    class Backend:
        def index_exists(self, _namespace):
            return True

    monkeypatch.setattr("evaluation.benchmark.HybridRetriever", FakeHybridRetriever)
    build_retriever(
        "hybrid-rerank",
        documents,
        vector_backend=Backend(),
        reranker_model="test-reranker",
    )
    assert captured["namespace"] == RAG_VECTOR_NAMESPACE
    assert captured["reranker_model"] == "test-reranker"


def test_benchmark_does_not_mislabel_unavailable_reranking_or_full_pipeline(document):
    documents = [document("billing policy", source="guide.md")]

    class Backend:
        def index_exists(self, _namespace):
            return True

        def search(self, *_args, **_kwargs):
            return [ScoredDocument(documents[0], 1.0)]

    with pytest.raises(ValueError, match="hybrid-rerank is unavailable"):
        build_retriever("hybrid-rerank", documents, vector_backend=Backend())
    with pytest.raises(ValueError, match="production ChatbotEngine pipeline"):
        build_retriever("full", documents, vector_backend=Backend())


def test_standalone_answerer_does_not_include_retrieval_context():
    class Model:
        def __init__(self):
            self.messages = None

        def chat(self, messages, *, model):
            self.messages = messages
            return type("Result", (), {"content": "standalone answer"})()

    model = Model()
    answer, citations = ChatModelAnswerer(model, "test-model").answer(
        "What is the limit?", [type("Document", (), {"page_content": "hidden corpus context"})()]
    )

    assert answer == "standalone answer"
    assert citations == []
    assert "hidden corpus context" not in model.messages[0]["content"]
    assert "general knowledge" in model.messages[0]["content"]
    assert "Question: What is the limit?" in model.messages[0]["content"]


def test_full_engine_records_pipeline_time_instead_of_generation_time():
    class Engine:
        async def chat(self, question):
            return type(
                "Response",
                (),
                {
                    "result": "pipeline answer",
                    "sources": [type("Source", (), {"source": "guide.md", "snippet": "evidence"})()],
                    "citations": [type("Citation", (), {"source_name": "guide.md"})()],
                },
            )()

    example = load_examples(Path("evaluation/dataset/datapizza"), "dev")[0]
    result = run_end_to_end(
        [example],
        None,
        EngineAnswerer(Engine()),
        configuration="full",
        split="dev",
        top_k=5,
    )

    timing = result["records"][0]["timing"]
    assert timing["pipeline_seconds"] >= 0
    assert "generation_seconds" not in timing
    assert "retrieval_latency_p50_seconds" not in result["summary"]
    assert result["records"][0]["context_available"] is False
    assert result["records"][0]["metrics"]["evidence_availability"] is None


def test_abstention_summary_uses_category_denominators():
    examples = [
        BenchmarkExample("answerable", "known", True, "known", ["guide.md"], ["known"]),
        BenchmarkExample("unanswerable", "unknown", False, None, [], []),
    ]

    class Answerer:
        def answer(self, question, _documents):
            return ("The answer is known." if question == "known" else "I do not have enough information."), []

    result = run_end_to_end(examples, None, Answerer(), configuration="standalone", split="dev", top_k=1)
    summary = result["summary"]
    assert summary["answerable_evaluated_count"] == 1
    assert summary["unanswerable_evaluated_count"] == 1
    assert summary["correct_abstention_rate"] == 1.0
    assert summary["unsupported_answer_rate"] == 0.0
    assert summary["incorrect_refusal_rate"] == 0.0


def test_failed_questions_are_counted_without_fake_quality_metrics():
    example = BenchmarkExample("failed", "question", True, "answer", ["guide.md"], ["answer"])

    class FailingRetriever:
        def retrieve(self, _question, _limit):
            raise RuntimeError("backend unavailable")

    result = run_retrieval([example], FailingRetriever(), configuration="dense", split="dev", top_k=1)
    assert result["summary"]["failed_count"] == 1
    assert result["summary"]["successful_count"] == 0
    assert "recall_at_k" not in result["summary"]


def test_benchmark_checkpoint_resume_preserves_completed_records(tmp_path):
    examples = [
        BenchmarkExample("q1", "one", True, "one", ["guide.md"], []),
        BenchmarkExample("q2", "two", True, "two", ["guide.md"], []),
    ]

    class Retriever:
        def retrieve(self, question, _limit):
            return [type("Doc", (), {"metadata": {"source": "guide.md"}})()]

    first = run_retrieval(
        examples[:1], Retriever(), configuration="bm25", split="dev", top_k=1, checkpoint_dir=tmp_path
    )
    assert len(first["records"]) == 1
    resumed = run_retrieval(
        examples,
        Retriever(),
        configuration="bm25",
        split="dev",
        top_k=1,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert [record["question_id"] for record in resumed["records"]] == ["q1", "q2"]
    assert len(tmp_path.joinpath("records.jsonl").read_text(encoding="utf-8").splitlines()) == 2
