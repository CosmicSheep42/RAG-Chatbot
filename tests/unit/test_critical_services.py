from types import SimpleNamespace

from langchain_core.documents import Document

from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.rag.context_assembler import DocumentContextAssembler
from local_rag_chatbot.rag.index_manager import IndexManager, RAGIndexStatus
from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceEstimator, RetrievalDecision
from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy
from local_rag_chatbot.rag.retrieval_service import RetrievalService
from local_rag_chatbot.rag.chatbot import RAGChatbot
from local_rag_chatbot.rag.answer_service import AnswerService


def _policy() -> RetrievalPolicy:
    return RetrievalPolicy(
        intent=QueryIntent.FACTUAL_LOOKUP,
        retrieval_limit=1,
        context_limit=1,
        expand_by_source=False,
        max_sources=1,
        source_chunk_limit=1,
        coverage_min_chunks=1,
        coverage_min_ratio=0.0,
        retry_retrieval_limit=2,
        retry_context_limit=2,
    )


def test_context_builder_preserves_source_metadata_and_deduplicates_citations():
    document = Document(
        "The default tenant limit is 120 requests per minute.",
        metadata={"source": "platform-overview.md", "document_id": "platform", "chunk_id": "c1"},
    )
    builder = ContextBuilder()

    context = builder.format_context([document])
    citations = builder.build_citations([document, document])

    assert "Source file: platform-overview.md" in context
    assert "<UNTRUSTED_DOCUMENT_CONTENT>" in context
    assert len(citations) == 1
    assert citations[0]["source_name"] == "platform-overview.md"
    assert 'reference="S1"' in context


def test_context_references_and_citations_share_stable_ids():
    documents = [
        Document("first", metadata={"source": "guide.md", "document_id": "d", "chunk_id": "1"}),
        Document("first", metadata={"source": "guide.md", "document_id": "d", "chunk_id": "1"}),
        Document("second", metadata={"source": "guide.md", "document_id": "d", "chunk_id": "2"}),
    ]
    builder = ContextBuilder()

    context = builder.format_context(documents)
    citations = builder.build_citations(documents)

    assert context.count("reference=") == 2
    assert [citation["citation_id"] for citation in citations] == ["S1", "S2"]


def test_context_budget_skips_oversized_document_and_keeps_later_evidence():
    builder = ContextBuilder(max_context_tokens=80, reserved_prompt_tokens=20)
    documents = [
        Document("x" * 1000, metadata={"source": "large.md", "chunk_id": "large"}),
        Document("short useful evidence", metadata={"source": "small.md", "chunk_id": "small"}),
    ]

    context = builder.format_context(documents)
    citations = builder.build_citations(documents)

    assert "large.md" not in context
    assert "small.md" in context
    assert [citation["source"] for citation in citations] == ["small.md"]


def test_confidence_is_based_on_context_that_generation_receives():
    large = Document(
        "x" * 1000,
        metadata={"source": "large.md", "chunk_id": "large", "context_role": "hit", "retrieval_dense_score": 0.95},
    )
    small = Document("unrelated", metadata={"source": "small.md", "chunk_id": "small"})
    builder = ContextBuilder(max_context_tokens=80, reserved_prompt_tokens=20)
    coverage = SimpleNamespace(sufficient=True, score=1.0, dominant_source="small.md", reason="ok")

    class Assembler:
        def evaluate_coverage(self, documents, _policy):
            return (
                coverage
                if documents
                else SimpleNamespace(sufficient=False, score=0.0, dominant_source=None, reason="empty")
            )

    chatbot = object.__new__(RAGChatbot)
    chatbot.context_builder = builder
    chatbot.index_manager = SimpleNamespace(
        context_assembler=Assembler(),
        confidence_estimator=RetrievalConfidenceEstimator(),
    )

    selected, _coverage, assessment = chatbot._select_and_assess_final_context("question", [large, small], _policy())

    assert [document.metadata["source"] for document in selected] == ["small.md"]
    assert assessment.decision is RetrievalDecision.REFUSE


def test_invalid_citation_references_are_removed():
    document = Document("supported", metadata={"source": "guide.md", "document_id": "d", "chunk_id": "1"})
    service = object.__new__(AnswerService)
    service.context_builder = ContextBuilder()

    assert service.validate_citations("Fact [S1]. Fake [S99].", [document]) == "Fact [S1]. Fake ."


def test_retrieval_service_retries_and_keeps_the_better_result():
    first = Document("weak", metadata={"retrieval_score": 0.1})
    second = Document("strong", metadata={"retrieval_score": 0.9})

    class Retriever:
        calls = 0

        def retrieve(self, query, **kwargs):
            self.calls += 1
            return [first] if self.calls == 1 else [second]

    class Assembler:
        def expand_context(self, query, documents, policy):
            score = documents[0].page_content == "strong"
            return documents, SimpleNamespace(sufficient=score, score=0.8 if score else 0.1)

    class Estimator:
        def assess(self, documents, coverage, policy):
            score = 0.8 if documents[0].page_content == "strong" else 0.1
            decision = RetrievalDecision.ANSWER if score > 0.5 else RetrievalDecision.RETRY
            return SimpleNamespace(score=score, decision=decision, reason_codes=())

    result = RetrievalService(
        retriever=Retriever(),
        context_assembler=Assembler(),
        confidence_estimator=Estimator(),
    ).retrieve("question", _policy())

    assert [document.page_content for document in result.documents] == ["strong"]
    assert result.metadata["retried"] is True


def test_retrieval_service_returns_empty_result_without_retrying():
    class Retriever:
        def retrieve(self, query, **kwargs):
            return []

    class Assembler:
        def expand_context(self, query, documents, policy):
            return [], SimpleNamespace(sufficient=False, score=0.0)

    class Estimator:
        def assess(self, documents, coverage, policy):
            return SimpleNamespace(score=0.0, decision=RetrievalDecision.REFUSE, reason_codes=())

    result = RetrievalService(
        retriever=Retriever(),
        context_assembler=Assembler(),
        confidence_estimator=Estimator(),
    ).retrieve("missing", _policy())

    assert result.documents == ()
    assert result.metadata["retried"] is False


def test_context_assembler_preserves_order_and_reports_coverage():
    documents = [
        Document("before", metadata={"source": "guide.md", "chunk_index": 0}),
        Document("answer", metadata={"source": "guide.md", "chunk_index": 1}),
        Document("after", metadata={"source": "guide.md", "chunk_index": 2}),
    ]
    selected, coverage = DocumentContextAssembler(documents).expand_context("answer", [documents[1]], _policy())

    assert [document.page_content for document in selected] == ["answer"]
    assert coverage.selected_chunks == 1
    assert coverage.dominant_source == "guide.md"


def test_index_manager_marks_empty_corpus_and_failures_without_partial_state():
    manager = object.__new__(IndexManager)
    manager.documents = []
    manager.retriever = None
    manager.context_assembler = None
    manager.retrieval_service = None
    manager.rag_status = RAGIndexStatus.NOT_INITIALIZED
    manager.rag_error = None
    manager._load_documents = lambda: []

    assert manager.prepare() is True
    assert manager.rag_status == RAGIndexStatus.EMPTY
    assert manager.has_empty_corpus

    manager._load_documents = lambda: (_ for _ in ()).throw(RuntimeError("index failed"))
    try:
        manager.prepare()
    except RuntimeError as error:
        assert str(error) == "index failed"
    else:
        raise AssertionError("prepare should propagate indexing failures")
    assert manager.rag_status == RAGIndexStatus.ERROR
    assert manager.rag_error == "index failed"


def test_retry_decision_is_not_generation_permission():
    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(retrieval_confidence_threshold=0.5)
    retry = SimpleNamespace(confidence=SimpleNamespace(decision=RetrievalDecision.RETRY), confidence_score=0.9)

    assert chatbot._should_refuse_retrieval(retry)
