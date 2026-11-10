import asyncio
from types import SimpleNamespace

import pytest

from local_rag_chatbot.conversation.query_rewriter import QueryRewriteResult
from local_rag_chatbot.engine import ChatbotEngine
from local_rag_chatbot.request_context import RequestContext


@pytest.mark.asyncio
async def test_same_conversation_turns_are_serialized():
    active = 0
    maximum_active = 0
    remembered: list[tuple[str, str]] = []

    class InputPipeline:
        def process_for_routing(self, query):
            return query

        def process_for_rag(self, query):
            return query

    class Router:
        def route(self, query, _allow_llm):
            return {"route": "llm", "reason": "test"}

    class Rewriter:
        async def arewrite(self, query, _history):
            return QueryRewriteResult(query, query, False, "disabled")

    class Store:
        def get_history(self, _conversation_id):
            return []

        def append(self, conversation_id, role, content):
            remembered.append((conversation_id, role))

    class Chatbot:
        async def achat(self, query, *_args, **_kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"result": query, "source_documents": []}

    services = SimpleNamespace(
        input_pipeline=InputPipeline(),
        router=Router(),
        query_rewriter=Rewriter(),
        conversation_store=Store(),
        chatbot=Chatbot(),
        context_builder=SimpleNamespace(build_citations=lambda _documents: []),
    )
    engine = ChatbotEngine.__new__(ChatbotEngine)
    engine._services = services
    engine.config = SimpleNamespace(conversation_max_sessions=10)
    engine._reindexing = False
    engine._active_rag_queries = 0
    engine._rag_state_lock = asyncio.Lock()
    engine._no_active_rag_queries = asyncio.Event()
    engine._no_active_rag_queries.set()

    await asyncio.gather(
        engine._chat_async("one", "same", RequestContext("1", "same")),
        engine._chat_async("two", "same", RequestContext("2", "same")),
    )

    assert maximum_active == 1
    assert remembered.count(("same", "user")) == 2
