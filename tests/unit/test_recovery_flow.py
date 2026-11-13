import asyncio
from types import SimpleNamespace
from threading import Event

import pytest
from langchain_core.documents import Document

from local_rag_chatbot.rag.chatbot import RAGChatbot
from local_rag_chatbot.rag.index_manager import RAGIndexStatus
from local_rag_chatbot.rag.retrieval_confidence import RetrievalDecision
from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy
from local_rag_chatbot.rag.retrieval_result import RetrievalResult


def _policy() -> RetrievalPolicy:
    return RetrievalPolicy(QueryIntent.FACTUAL_LOOKUP, 2, 2, False, 1, 2, 1, 0.0, 4, 4)


def _assessment(decision, score):
    return SimpleNamespace(
        decision=decision,
        score=score,
        confidence=score,
        reason_codes=(),
        as_debug_payload=lambda: {"decision": decision.value, "confidence": score},
    )


def _chatbot(initial, final, answer="grounded"):
    policy = _policy()
    source = Document("evidence", metadata={"source": "guide.md", "chunk_id": "1", "context_role": "hit"})
    recovered = Document("recovered", metadata={"source": "guide.md", "chunk_id": "2", "context_role": "hit"})
    coverage = SimpleNamespace(score=1.0, reason="ok", dominant_source="guide.md", sufficient=True)

    class Assembler:
        def expand_context(self, _query, documents, _policy):
            return list(documents), coverage

        def evaluate_coverage(self, _documents, _policy):
            return coverage

    class Estimator:
        calls = 0

        def assess(self, _documents, _coverage, _policy):
            self.calls += 1
            return initial if self.calls == 1 else final

    class Retrieval:
        def retrieve(self, *_args, **_kwargs):
            return RetrievalResult("question", (source,), (source,), initial, policy, coverage)

    chatbot = object.__new__(RAGChatbot)
    chatbot.settings = SimpleNamespace(
        tool_calling_enabled=True,
        tool_calling_max_rounds=1,
        retrieval_confidence_threshold=0.35,
    )
    chatbot.index_manager = SimpleNamespace(
        rag_status=RAGIndexStatus.INDEXED,
        retrieval_service=Retrieval(),
        context_assembler=Assembler(),
        confidence_estimator=Estimator(),
    )
    chatbot.context_builder = SimpleNamespace(select_documents=lambda documents: list(documents))
    chatbot.policy_resolver = SimpleNamespace(resolve=lambda _query: policy)

    async def generate(*_args, **_kwargs):
        return answer

    chatbot.answer_service = SimpleNamespace(agenerate=generate)

    async def recover(*_args, **_kwargs):
        return (
            "",
            [source, recovered],
            [
                {
                    "name": "search_documents",
                    "arguments": {"query": "question"},
                    "payload": {"status": "success_empty"},
                    "new_document_count": 0,
                }
            ],
            True,
        )

    chatbot._agenerate_answer_with_optional_tools = recover
    return chatbot


@pytest.mark.asyncio
async def test_recoverable_retry_is_reassessed_before_generation():
    chatbot = _chatbot(_assessment(RetrievalDecision.RETRY, 0.4), _assessment(RetrievalDecision.ANSWER, 0.9))

    result = await chatbot.aanswer("question")

    assert result["result"] == "grounded"
    assert result["refused"] is False
    assert result["tool_calling_used"] is True


@pytest.mark.asyncio
async def test_recoverable_refusal_attempts_bounded_recovery():
    chatbot = _chatbot(_assessment(RetrievalDecision.REFUSE, 0.3), _assessment(RetrievalDecision.ANSWER, 0.9))

    result = await chatbot.aanswer("question")

    assert result["refused"] is False
    assert result["tool_calling_used"] is True


@pytest.mark.asyncio
async def test_failed_recovery_refuses_without_generation():
    chatbot = _chatbot(_assessment(RetrievalDecision.RETRY, 0.4), _assessment(RetrievalDecision.RETRY, 0.4))
    generated = False

    async def should_not_generate(*_args, **_kwargs):
        nonlocal generated
        generated = True
        return "bad"

    chatbot.answer_service.agenerate = should_not_generate
    result = await chatbot.aanswer("question")

    assert result["refused"] is True
    assert generated is False
    assert result["tool_calls"][0]["payload"]["status"] == "success_empty"
    assert result["tool_calling_used"] is True
    assert chatbot.index_manager.confidence_estimator.calls == 2


def test_equivalent_punctuation_and_whitespace_queries_do_not_repeat_search():
    assert not RAGChatbot._needs_recovery_search(
        [
            {
                "name": "search_documents",
                "arguments": {"query": "  mass-cure   wounds? "},
                "payload": {"status": "success_empty"},
            }
        ],
        "mass cure wounds",
    )


@pytest.mark.asyncio
async def test_remaining_retry_runs_standalone_search_and_reassesses_evidence():
    chatbot = _chatbot(
        _assessment(RetrievalDecision.RETRY, 0.4),
        _assessment(RetrievalDecision.ANSWER, 0.9),
    )
    assessments = iter(
        [
            _assessment(RetrievalDecision.RETRY, 0.4),
            _assessment(RetrievalDecision.RETRY, 0.45),
            _assessment(RetrievalDecision.ANSWER, 0.9),
        ]
    )
    chatbot.index_manager.confidence_estimator.assess = lambda *_args: next(assessments)
    fallback_calls = []
    extra = Document("new grounded evidence", metadata={"source": "guide.md", "chunk_id": "3"})

    def search(query, top_k=4, request_context=None):
        fallback_calls.append((query, top_k))
        return "{}", [extra], {"status": "success"}

    chatbot.tool_registry = SimpleNamespace(execute=lambda _name, args, request_context=None: search(**args))

    async def recover(*_args, **_kwargs):
        return (
            "",
            [Document("irrelevant new heading", metadata={"source": "guide.md", "chunk_id": "2"})],
            [
                {
                    "name": "search_documents",
                    "arguments": {"query": "different query"},
                    "payload": {"status": "success"},
                    "new_document_count": 1,
                }
            ],
            True,
        )

    chatbot._agenerate_answer_with_optional_tools = recover
    result = await chatbot.aanswer("question")

    assert result["refused"] is False
    assert result["result"] == "grounded"
    assert fallback_calls == [("question", 4)]
    assert result["retrieval_confidence"] == 0.9
    assert len(result["tool_calls"]) == 2


def test_reassessment_keeps_new_retrieved_evidence_ahead_of_expansion_filler():
    chatbot = _chatbot(_assessment(RetrievalDecision.RETRY, 0.4), _assessment(RetrievalDecision.ANSWER, 0.9))
    filler = Document("expanded filler", metadata={"source": "guide.md", "chunk_id": "1"})
    initial = Document("initial retrieved evidence", metadata={"source": "guide.md", "chunk_id": "0"})
    recovered = Document("recovered answer evidence", metadata={"source": "guide.md", "chunk_id": "2"})
    coverage = chatbot.index_manager.context_assembler.evaluate_coverage([], _policy())
    chatbot.index_manager.context_assembler.expand_context = lambda *_args: ([filler], coverage)
    seen = []
    chatbot.index_manager.confidence_estimator.assess = lambda docs, *_args: (
        seen.extend(docs) or _assessment(RetrievalDecision.ANSWER, 0.9)
    )

    selected, _, assessment = chatbot._reassess_documents(
        "question",
        [initial, recovered],
        _policy(),
        preferred_documents=[recovered],
    )

    assert selected == [recovered, filler]
    assert seen == [recovered, filler]
    assert assessment.decision == RetrievalDecision.ANSWER


@pytest.mark.asyncio
async def test_cancelled_blocking_work_is_drained_before_task_exits():
    chatbot = object.__new__(RAGChatbot)
    started = Event()
    release = Event()

    def blocking_operation():
        started.set()
        release.wait(timeout=2)
        return "done"

    task = asyncio.create_task(chatbot._run_blocking(blocking_operation))
    while not started.is_set():
        await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
