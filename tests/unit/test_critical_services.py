from types import SimpleNamespace

from langchain_core.documents import Document

from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.rag.context_assembler import DocumentContextAssembler
from local_rag_chatbot.rag.index_manager import IndexManager, RAGIndexStatus
from local_rag_chatbot.rag.retrieval_confidence import RetrievalDecision
from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy
from local_rag_chatbot.rag.retrieval_service import RetrievalService


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
