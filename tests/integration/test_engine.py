import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from local_rag_chatbot.conversation.query_rewriter import QueryRewriteResult
from local_rag_chatbot.engine import ChatbotEngine
from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.request_context import RequestContext


@pytest.mark.asyncio
async def test_engine_composes_routing_retrieval_generation_and_citations():
    source = Document(
        "The default tenant limit is 120 requests per minute.",
        metadata={"source": "platform-overview.md", "document_id": "platform", "chunk_id": "c1"},
    )

    class InputPipeline:
        def process_for_routing(self, query):
            return query

        def process_for_rag(self, query):
            return query

    class Router:
        def route(self, query, _allow_llm):
            return {"route": "llm", "reason": "no shortcut"}

    class Rewriter:
        async def arewrite(self, query, _history):
            return QueryRewriteResult(query, query, False, "disabled")

    class Chatbot:
        async def achat(self, query, *_args, **_kwargs):
            return {"result": "120 requests per minute", "source_documents": [source]}

    services = SimpleNamespace(
        input_pipeline=InputPipeline(),
        router=Router(),
        query_rewriter=Rewriter(),
        conversation_store=None,
        chatbot=Chatbot(),
        context_builder=ContextBuilder(),
    )
    engine = ChatbotEngine.__new__(ChatbotEngine)
    engine._services = services
    engine._initialized = True
    engine._reindexing = False
    engine._active_rag_queries = 0
    engine._rag_state_lock = asyncio.Lock()
    engine._no_active_rag_queries = asyncio.Event()
    engine._no_active_rag_queries.set()

    response = await engine._chat_async("What is the limit?", None, RequestContext("req-1", None))

    assert response.result == "120 requests per minute"
    assert response.citations[0].source_name == "platform-overview.md"
