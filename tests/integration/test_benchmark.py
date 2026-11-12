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
    validate_configuration_mode,
    _configuration_fingerprint,
    _configuration_snapshot,
    _load_checkpoint_records,
    _prepare_run_output,
    _run_live,
)
from local_rag_chatbot.rag.index_manager import RAG_VECTOR_NAMESPACE
from local_rag_chatbot.vectorstores.contracts import ScoredDocument


def test_bm25_benchmark_smoke_produces_structured_results(tmp_path):
    corpus = load_corpus(Path("knowledge-base"))
    examples = load_examples(Path("evaluation/dataset/datapizza"), "dev")[:2]

    retrieval = run_retrieval(examples, BM25Retriever(corpus), configuration="bm25", split="dev", top_k=3)
    assert retrieval["records"]
    assert retrieval["records"][0]["status"] == "success"
    assert "retrieval_sparse_score" in retrieval["records"][0]["retrieved_documents"][0]
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
    class Document:
        page_content = "selected evidence"
        metadata = {"source": "guide.md", "source_char_start": 4, "source_char_end": 21}

    class Chatbot:
        def __init__(self):
            class RetrievalService:
                def retrieve(self, *_args, **_kwargs):
                    return type("Result", (), {"documents": [Document()]})()

            self.index_manager = type(
                "IndexManager",
                (),
                {"retrieval_service": RetrievalService()},
            )()

        async def achat(self, *_args, **_kwargs):
            self.index_manager.retrieval_service.retrieve("question", None)
            return {
                "source_documents": [Document()],
                "retrieval_confidence_assessment": {"decision": "answer", "confidence": 0.9},
                "refused": False,
                "partial_context": True,
                "tool_calls": [{"name": "search_documents"}],
            }

    class Engine:
        _services = type("Services", (), {"chatbot": Chatbot()})()

        async def chat(self, question):
            await self._services.chatbot.achat(question)
            return type(
                "Response",
                (),
                {
                    "result": "pipeline answer [S1]",
                    "sources": [type("Source", (), {"source": "guide.md", "snippet": "evidence"})()],
                    "citations": [type("Citation", (), {"source_name": "guide.md", "citation_id": "S1"})()],
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
    assert result["records"][0]["context_available"] is True
    assert result["records"][0]["metrics"]["evidence_availability"] is None
    assert result["records"][0]["selected_context"] == ["selected evidence"]
    assert result["records"][0]["pipeline_trace"]["retrieval_confidence_assessment"]["confidence"] == 0.9
    assert result["records"][0]["pipeline_trace"]["selected_evidence"][0]["document_id"] == "guide.md"
    assert (
        result["records"][0]["pipeline_trace"]["retrieval_attempts"][0]["documents"][0]["content"]
        == "selected evidence"
    )
    assert result["records"][0]["citation_identifiers"] == ["S1"]
    json.dumps(result, allow_nan=False)


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


def test_retrieval_only_summary_does_not_report_end_to_end_latency(document):
    result = run_retrieval(
        [BenchmarkExample("q1", "one", True, "one", ["guide.md"], [])],
        type("Retriever", (), {"retrieve": lambda self, _question, _limit: [document("one", source="guide.md")]})(),
        configuration="bm25",
        split="dev",
        top_k=1,
    )
    assert "retrieval_latency_p50_seconds" in result["summary"]
    assert "end_to_end_latency_p50_seconds" not in result["summary"]


@pytest.mark.parametrize(
    ("configuration", "mode", "supported"),
    [
        ("standalone", "retrieval-only", False),
        ("standalone", "end-to-end", True),
        ("bm25", "retrieval-only", True),
        ("bm25", "end-to-end", True),
        ("dense", "retrieval-only", True),
        ("dense", "end-to-end", True),
        ("hybrid", "retrieval-only", True),
        ("hybrid", "end-to-end", True),
        ("hybrid-rerank", "retrieval-only", True),
        ("hybrid-rerank", "end-to-end", True),
        ("full", "retrieval-only", False),
        ("full", "end-to-end", True),
    ],
)
def test_configuration_mode_matrix(configuration, mode, supported):
    if supported:
        validate_configuration_mode(configuration, mode)
    else:
        with pytest.raises(ValueError):
            validate_configuration_mode(configuration, mode)


def test_bm25_compose_service_uses_live_dispatch_without_ollama_dependency():
    compose = Path("docker-compose.benchmark.yml").read_text(encoding="utf-8")
    bm25_block = compose.split("  benchmark-bm25:\n", 1)[1].split("  benchmark-bm25-e2e:", 1)[0]
    assert 'command: ["--live", "--configuration", "bm25"' in bm25_block
    assert "depends_on:" not in bm25_block
    bm25_e2e_block = compose.split("  benchmark-bm25-e2e:", 1)[1].split("  benchmark-dense:", 1)[0]
    assert 'command: ["--live", "--configuration", "bm25", "--mode", "end-to-end"' in bm25_e2e_block
    assert "depends_on: [ollama]" in bm25_e2e_block


def test_resume_rejects_changed_generation_model_before_execution(tmp_path, monkeypatch):
    args = type(
        "Args",
        (),
        {
            "dataset": Path("evaluation/dataset/datapizza"),
            "corpus": Path("knowledge-base"),
            "split": "dev",
            "configuration": "bm25",
            "end_to_end": True,
            "top_k": 5,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "output": tmp_path / "run",
            "resume": False,
            "run_id": "test",
        },
    )()
    snapshot = _configuration_snapshot(args)
    args.output.mkdir()
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "run_status": "running",
                "configuration_fingerprint": _configuration_fingerprint(snapshot),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERATION_MODEL", "different-model")
    args.resume = True
    with pytest.raises(ValueError, match="configuration mismatch"):
        _prepare_run_output(args)


def test_corrupt_checkpoint_is_rejected(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text('{"question_id":"q1"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        _load_checkpoint_records(tmp_path)


def test_bm25_live_end_to_end_uses_generation_without_embeddings(monkeypatch, tmp_path):
    calls = {"chat": 0}

    class FakeChatModel:
        def chat(self, _messages, *, model):
            calls["chat"] += 1
            return type("Result", (), {"content": f"answer from {model}"})()

    monkeypatch.setattr("local_rag_chatbot.providers.llm.create_chat_model", lambda _settings: FakeChatModel())
    monkeypatch.setattr(
        "local_rag_chatbot.providers.embeddings.create_embeddings",
        lambda _settings: (_ for _ in ()).throw(AssertionError("BM25 must not create embeddings")),
    )
    args = type(
        "Args",
        (),
        {
            "dataset": Path("evaluation/dataset/datapizza"),
            "corpus": Path("knowledge-base"),
            "split": "dev",
            "configuration": "bm25",
            "end_to_end": True,
            "top_k": 2,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "data_dir": tmp_path,
            "llm_provider": None,
            "embedding_provider": None,
            "generation_model": "test-model",
            "embedding_model": None,
            "vector_store_provider": None,
            "reranker_model": None,
            "reranker_device": None,
            "checkpoint_dir": None,
            "resume": False,
            "retry_failed": False,
        },
    )()
    result = _run_live(args, load_examples(args.dataset, "dev")[:1])
    assert calls["chat"] == 1
    assert result["metadata"]["configuration"] == "bm25"
    assert result["metadata"]["experiment_mode"] == "end-to-end"
