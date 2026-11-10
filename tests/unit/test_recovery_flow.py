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
        return "", [source, recovered], [{"name": "search_documents"}], True

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
