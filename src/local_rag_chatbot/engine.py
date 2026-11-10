from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from local_rag_chatbot.application import ApplicationServices, create_application_services
from local_rag_chatbot.config import ChatbotConfig
from local_rag_chatbot.models import ChatResponse, Citation, SourceSnippet, ToolCall
from local_rag_chatbot.observability import metrics
from local_rag_chatbot.rag.contracts import ChatModel, ConversationStore
from local_rag_chatbot.request_context import RequestContext

logger = logging.getLogger(__name__)


class ChatbotEngine:
    """Public programmatic API for the local RAG chatbot.

    `initialize()` and `shutdown()` are idempotent. The engine does not depend on
    FastAPI and can be used from scripts, workers, tests, or HTTP adapters.
    """

    def __init__(
        self,
        *,
        config: ChatbotConfig,
        chat_model: ChatModel | None = None,
        embeddings: object | None = None,
        conversation_store: ConversationStore | None = None,
        services: ApplicationServices | None = None,
    ) -> None:
        self.config = config
        self._services = services or create_application_services(
            config,
            chat_model=chat_model,
            embeddings=embeddings,
            conversation_store=conversation_store,
        )
        self._initialized = False
        self._initializing = False
        self._init_error: str | None = None
        self._reindexing = False
        self._active_rag_queries = 0
        self._rag_state_lock = asyncio.Lock()
        self._no_active_rag_queries = asyncio.Event()
        self._no_active_rag_queries.set()

    @property
    def services(self) -> ApplicationServices:
        return self._services

    @property
    def ready(self) -> bool:
        return (
            self._initialized
            and not self._reindexing
            and self._services.index_manager.rag_status.value not in {"not_initialized", "error"}
        )

    def status(self) -> dict:
        retriever = self._services.index_manager.retriever
        if self._initializing:
            fallback_reranking_status = "initializing"
        elif self.config.reranking_mode == "false":
            fallback_reranking_status = "disabled"
        else:
            fallback_reranking_status = "not_applicable"
        reranking_status = (
            retriever.reranking_status
            if retriever is not None
            else {
                "configured_mode": self.config.reranking_mode,
                "active": False,
                "status": fallback_reranking_status,
                "reason": None,
            }
        )
        return {
            "ready": self.ready,
            "initializing": self._initializing,
            "reindexing": self._reindexing,
            "error": self._init_error,
            "rag": self._services.chatbot.rag_status_payload(),
            "features": {
                "memory": self.config.conversation_memory_enabled,
                "query_rewriting": self.config.query_rewriting_enabled,
                "tool_calling": self.config.tool_calling_enabled,
                "reranking": reranking_status,
            },
            "controls": self.config.runtime_control_config(),
        }

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._initializing:
            while self._initializing:
                await asyncio.sleep(0.05)
            if self._initialized:
                return
        self._initializing = True
        self._init_error = None
        try:
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True
        except Exception as exc:
            self._init_error = str(exc)
            raise
        finally:
            self._initializing = False

    def _initialize_sync(self) -> None:
        self._services.router.setup()
        if not self._services.index_manager.load_or_prepare(auto_prepare=self.config.auto_index_on_startup):
            raise RuntimeError(
                "RAG index is missing or incompatible. Run 'python -m local_rag_chatbot.ingest' "
                "or set AUTO_INDEX_ON_STARTUP=true."
            )

    async def chat(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        session_id: str | None = None,
        request_context: RequestContext | None = None,
    ) -> ChatResponse:
        if not self._initialized:
            raise RuntimeError("ChatbotEngine is not initialized.")
        if self._reindexing:
            raise RuntimeError("ChatbotEngine is reindexing; RAG queries are temporarily unavailable.")
        from local_rag_chatbot.models import resolve_conversation_aliases

        effective_conversation_id = resolve_conversation_aliases(
            conversation_id=conversation_id,
            session_id=session_id,
        )
        context_conversation_id = request_context.conversation_id if request_context else None
        effective_conversation_id = resolve_conversation_aliases(
            conversation_id=effective_conversation_id,
            session_id=context_conversation_id,
        )
        context = request_context or RequestContext(request_id=str(uuid4()), conversation_id=effective_conversation_id)
        if request_context is not None and request_context.conversation_id != effective_conversation_id:
            context = RequestContext(
                request_id=request_context.request_id,
                conversation_id=effective_conversation_id,
            )
        return await self._chat_async(query, effective_conversation_id, context)

    async def reindex(self) -> None:
        if not self._initialized:
            raise RuntimeError("ChatbotEngine is not initialized.")
        await self._begin_reindexing()
        previous_error = self._init_error
        self._init_error = None
        try:
            await asyncio.to_thread(self._services.index_manager.prepare)
            self._init_error = None
        except Exception as exc:
            if self._services.index_manager.has_indexed_documents:
                self._init_error = previous_error
            else:
                self._init_error = str(exc)
            raise
        finally:
            await self._end_reindexing()

    def _chat_sync(self, query: str, conversation_id: str | None, context: RequestContext) -> ChatResponse:
        if self._reindexing:
            raise RuntimeError("ChatbotEngine is reindexing; RAG queries are temporarily unavailable.")
        cleaned_query = self._services.input_pipeline.process_for_routing(query)
        if not cleaned_query:
            raise ValueError("Query is empty after preprocessing.")

        route_result = self._services.router.route(cleaned_query, True)
        if route_result["route"] != "llm":
            self._remember(conversation_id, query, route_result["result"])
            return ChatResponse(
                result=route_result["result"],
                route=route_result["route"],
                reason=route_result["reason"],
                matched_category=route_result.get("matched_category"),
                matched_intent=route_result.get("matched_intent"),
                score=route_result.get("score"),
            )

        rewrite_result = self._rewrite_query(query, conversation_id)
        standalone_query = self._services.input_pipeline.process_for_rag(rewrite_result.standalone_query)
        original_query = rewrite_result.original_query.strip()
        if not standalone_query or not original_query:
            raise ValueError("Query is empty after preprocessing.")

        result = self._services.chatbot.chat(
            standalone_query,
            True,
            original_query=original_query,
            request_context=context,
        )
        if result is None:
            raise RuntimeError("Error processing query.")

        self._remember(conversation_id, query, result.get("result", ""))
        citations = self._services.context_builder.build_citations(result.get("source_documents", []))
        sources = [
            SourceSnippet(
                source=(getattr(document, "metadata", {}) or {}).get("source", "unknown"),
                snippet=(document.page_content[:300] if hasattr(document, "page_content") else str(document)[:300]),
            )
            for document in result.get("source_documents", [])
        ]
        return ChatResponse(
            result=result.get("result") or "",
            sources=sources,
            citations=[Citation.model_validate(citation) for citation in citations],
            tool_calls=[ToolCall.model_validate(call) for call in result.get("tool_calls", [])],
            tool_calling_used=result.get("tool_calling_used", False),
            route=route_result["route"],
            reason=route_result["reason"],
            matched_category=route_result.get("matched_category"),
            matched_intent=route_result.get("matched_intent"),
            score=route_result.get("score"),
            original_query=result.get("original_query"),
            standalone_query=result.get("standalone_query"),
            query_rewritten=result.get("query_rewritten", False),
        )

    async def _chat_async(self, query: str, conversation_id: str | None, context: RequestContext) -> ChatResponse:
        cleaned_query = self._services.input_pipeline.process_for_routing(query)
        if not cleaned_query:
            raise ValueError("Query is empty after preprocessing.")

        route_result = await asyncio.to_thread(self._services.router.route, cleaned_query, True)
        if route_result["route"] != "llm":
            self._remember(conversation_id, query, route_result["result"])
            return ChatResponse(
                result=route_result["result"],
                route=route_result["route"],
                reason=route_result["reason"],
                matched_category=route_result.get("matched_category"),
                matched_intent=route_result.get("matched_intent"),
                score=route_result.get("score"),
            )

        async with self._rag_query_slot():
            rewrite_result = await self._arewrite_query(query, conversation_id)
            standalone_query = self._services.input_pipeline.process_for_rag(rewrite_result.standalone_query)
            original_query = rewrite_result.original_query.strip()
            if not standalone_query or not original_query:
                raise ValueError("Query is empty after preprocessing.")

            result = await self._services.chatbot.achat(
                standalone_query,
                True,
                original_query=original_query,
                request_context=context,
            )
        if result is None:
            raise RuntimeError("Error processing query.")

        self._remember(conversation_id, query, result.get("result", ""))
        citations = self._services.context_builder.build_citations(result.get("source_documents", []))
        sources = [
            SourceSnippet(
                source=(getattr(document, "metadata", {}) or {}).get("source", "unknown"),
                snippet=(document.page_content[:300] if hasattr(document, "page_content") else str(document)[:300]),
            )
            for document in result.get("source_documents", [])
        ]
        return ChatResponse(
            result=result.get("result") or "",
            sources=sources,
            citations=[Citation.model_validate(citation) for citation in citations],
            tool_calls=[ToolCall.model_validate(call) for call in result.get("tool_calls", [])],
            tool_calling_used=result.get("tool_calling_used", False),
            route=route_result["route"],
            reason=route_result["reason"],
            matched_category=route_result.get("matched_category"),
            matched_intent=route_result.get("matched_intent"),
            score=route_result.get("score"),
            original_query=result.get("original_query"),
            standalone_query=result.get("standalone_query"),
            query_rewritten=result.get("query_rewritten", False),
        )

    def _rewrite_query(self, query: str, conversation_id: str | None):
        memory = self._services.conversation_store
        history = memory.get_history(conversation_id) if memory is not None and conversation_id is not None else []
        try:
            with metrics.timer("query_rewriting_seconds"):
                return self._services.query_rewriter.rewrite(query, history)
        except Exception as exc:
            logger.warning("Query rewriting failed; using original query")
            from local_rag_chatbot.conversation.query_rewriter import QueryRewriteResult

            return QueryRewriteResult(
                original_query=query,
                standalone_query=query,
                rewritten=False,
                method="fallback",
                error=str(exc),
            )

    async def _arewrite_query(self, query: str, conversation_id: str | None):
        memory = self._services.conversation_store
        history = memory.get_history(conversation_id) if memory is not None and conversation_id is not None else []
        try:
            with metrics.timer("query_rewriting_seconds"):
                return await self._services.query_rewriter.arewrite(query, history)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Query rewriting failed; using original query")
            from local_rag_chatbot.conversation.query_rewriter import QueryRewriteResult

            return QueryRewriteResult(
                original_query=query,
                standalone_query=query,
                rewritten=False,
                method="fallback",
                error=str(exc),
            )

    def _remember(self, conversation_id: str | None, user_query: str, assistant_result: str) -> None:
        memory = self._services.conversation_store
        if memory is None or conversation_id is None:
            return
        memory.append(conversation_id, "user", user_query)
        memory.append(conversation_id, "assistant", assistant_result)

    async def shutdown(self) -> None:
        self._initialized = False
        self._initializing = False

    async def _begin_reindexing(self) -> None:
        async with self._rag_state_lock:
            if self._reindexing:
                raise RuntimeError("ChatbotEngine is already reindexing.")
            self._reindexing = True
            if self._active_rag_queries == 0:
                self._no_active_rag_queries.set()
            else:
                self._no_active_rag_queries.clear()
        await self._no_active_rag_queries.wait()

    async def _end_reindexing(self) -> None:
        async with self._rag_state_lock:
            self._reindexing = False

    def _rag_query_slot(self):
        engine = self

        class _RagQuerySlot:
            async def __aenter__(self) -> None:
                async with engine._rag_state_lock:
                    if engine._reindexing:
                        raise RuntimeError("ChatbotEngine is reindexing; RAG queries are temporarily unavailable.")
                    engine._active_rag_queries += 1
                    engine._no_active_rag_queries.clear()

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                async with engine._rag_state_lock:
                    engine._active_rag_queries = max(0, engine._active_rag_queries - 1)
                    if engine._active_rag_queries == 0:
                        engine._no_active_rag_queries.set()

        return _RagQuerySlot()


def create_default_engine(config: ChatbotConfig) -> ChatbotEngine:
    return ChatbotEngine(config=config)
