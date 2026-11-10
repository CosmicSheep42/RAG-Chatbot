from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Mapping, TYPE_CHECKING

from local_rag_chatbot.app_errors import map_provider_error
from local_rag_chatbot.documents.tool_config_repository import ToolCallingConfig, ToolDefinition
from local_rag_chatbot.input_cleaner import TextPreprocessorPipeline
from local_rag_chatbot.observability import metrics
from local_rag_chatbot.providers.errors import ProviderError
from local_rag_chatbot.rag.answer_service import AnswerService
from local_rag_chatbot.rag.context_builder import ContextBuilder
from local_rag_chatbot.rag.contracts import ChatModel
from local_rag_chatbot.rag.index_manager import IndexManager, RAGIndexStatus
from local_rag_chatbot.providers.llm.types import ToolCall, ToolSpec
from local_rag_chatbot.rag.retrieval_confidence import (
    RetrievalConfidenceAssessment,
    RetrievalDecision,
    RetrievalReason,
)
from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicy, RetrievalPolicyResolver
from local_rag_chatbot.rag.retrieval_result import RetrievalResult
from local_rag_chatbot.rag.tool_registry import ToolRegistry
from local_rag_chatbot.request_context import RequestContext
from local_rag_chatbot.settings import AppSettings

if TYPE_CHECKING:
    from local_rag_chatbot.rag.context_assembler import CoverageReport

logger = logging.getLogger(__name__)


class RAGChatbot:
    def __init__(
        self,
        *,
        settings: AppSettings,
        input_pipeline: TextPreprocessorPipeline,
        index_manager: IndexManager,
        policy_resolver: RetrievalPolicyResolver,
        answer_service: AnswerService,
        context_builder: ContextBuilder,
        tool_registry: ToolRegistry,
        tool_calling_config: ToolCallingConfig,
        chat_model: ChatModel,
    ) -> None:
        self.settings = settings
        self.input_pipeline = input_pipeline
        self.index_manager = index_manager
        self.policy_resolver = policy_resolver
        self.answer_service = answer_service
        self.context_builder = context_builder
        self.tool_registry = tool_registry
        self.tool_calling_config = tool_calling_config
        self.chat_model = chat_model
        self._register_tools()

    def load_existing_vectorstore(self) -> bool:
        return self.index_manager.load_existing()

    def setup_rag(self) -> bool:
        return self.index_manager.prepare()

    @property
    def retriever(self):
        return self.index_manager.retriever

    @property
    def rag_status(self):
        return self.index_manager.rag_status

    def rag_status_payload(self) -> dict[str, Any]:
        return self.index_manager.status()

    def chat(
        self,
        query: str,
        already_cleaned: bool = False,
        *,
        original_query: str | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        cleaned_query = query if already_cleaned else self.input_pipeline.process_for_rag(query)
        if not cleaned_query:
            return None
        return self.answer(
            cleaned_query,
            original_query=original_query,
            request_context=request_context,
        )

    async def achat(
        self,
        query: str,
        already_cleaned: bool = False,
        *,
        original_query: str | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        cleaned_query = query if already_cleaned else self.input_pipeline.process_for_rag(query)
        if not cleaned_query:
            return None
        return await self.aanswer(
            cleaned_query,
            original_query=original_query,
            request_context=request_context,
        )

    def answer(
        self,
        retrieval_query: str,
        *,
        original_query: str | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        generation_query = (original_query or retrieval_query).strip()
        retrieval_query = retrieval_query.strip()
        if self.index_manager.rag_status == RAGIndexStatus.EMPTY:
            return self._empty_corpus_response(generation_query, retrieval_query)

        if self.index_manager.retrieval_service is None:
            logger.warning(
                "RAG has not been initialized",
                extra=self._request_log_extra(request_context),
            )
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.aanswer(
                    retrieval_query,
                    original_query=original_query,
                    request_context=request_context,
                )
            )

        raise RuntimeError("RAGChatbot.answer() cannot run inside an active event loop; use await aanswer().")

    async def aanswer(
        self,
        retrieval_query: str,
        *,
        original_query: str | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        generation_query = (original_query or retrieval_query).strip()
        retrieval_query = retrieval_query.strip()
        if self.index_manager.rag_status == RAGIndexStatus.EMPTY:
            return self._empty_corpus_response(generation_query, retrieval_query)

        if self.index_manager.retrieval_service is None:
            logger.warning(
                "RAG has not been initialized",
                extra=self._request_log_extra(request_context),
            )
            return None

        try:
            with metrics.timer("retrieval_policy_seconds"):
                policy = self.policy_resolver.resolve(retrieval_query)
            retrieval_result = await self._run_blocking(
                self.index_manager.retrieval_service.retrieve,
                retrieval_query,
                policy,
                request_context=request_context,
            )
            documents = retrieval_result.documents
            coverage = retrieval_result.coverage
            policy = retrieval_result.policy or policy
            if coverage is None:
                raise RuntimeError("Retrieval did not produce coverage.")

            documents, coverage, confidence_assessment = await self._run_blocking(
                self._select_and_assess_final_context,
                retrieval_query,
                list(documents),
                policy,
            )
            confidence = confidence_assessment.score if confidence_assessment is not None else 0.0

            if confidence_assessment is None and confidence < self.settings.retrieval_confidence_threshold:
                return self._refusal_response(generation_query, retrieval_query, policy, None, coverage)

            partial_context = self._is_partial_enumeration_allowed(retrieval_result)
            recovery_only = (
                confidence_assessment is not None
                and confidence_assessment.decision in {RetrievalDecision.RETRY, RetrievalDecision.REFUSE}
                and self._is_recoverable_assessment(confidence_assessment)
            ) or (partial_context and self.settings.tool_calling_enabled)
            if recovery_only and not self.settings.tool_calling_enabled:
                return self._refusal_response(
                    generation_query, retrieval_query, policy, confidence_assessment, coverage
                )
            if (
                confidence_assessment is not None
                and confidence_assessment.decision == RetrievalDecision.REFUSE
                and not recovery_only
            ):
                return self._refusal_response(
                    generation_query, retrieval_query, policy, confidence_assessment, coverage
                )
            result_text, final_documents, tool_events, tool_used = await self._agenerate_answer_with_optional_tools(
                generation_query,
                documents,
                policy,
                standalone_query=retrieval_query,
                coverage=coverage,
                confidence_assessment=confidence_assessment,
                partial_context=partial_context,
                recovery_only=recovery_only,
                request_context=request_context,
            )
            if recovery_only:
                final_documents, final_coverage, final_confidence = await self._run_blocking(
                    self._reassess_documents,
                    retrieval_query,
                    final_documents,
                    policy,
                )
                final_partial = self._is_partial_enumeration_allowed(
                    RetrievalResult(
                        query=retrieval_query,
                        documents=tuple(final_documents),
                        candidates=tuple(final_documents),
                        confidence=final_confidence,
                        policy=policy,
                        coverage=final_coverage,
                    )
                )
                if final_confidence is None or (
                    final_confidence.decision != RetrievalDecision.ANSWER and not final_partial
                ):
                    return self._refusal_response(
                        generation_query, retrieval_query, policy, final_confidence, final_coverage
                    )
                result_text = await self.answer_service.agenerate(
                    generation_query,
                    list(final_documents),
                    policy,
                    coverage=final_coverage,
                    partial_context=final_partial,
                )
                documents = tuple(final_documents)
                coverage = final_coverage
                confidence_assessment = final_confidence
                confidence = final_confidence.score
                partial_context = final_partial
            return {
                "result": result_text,
                "source_documents": list(final_documents),
                "retrieval_confidence": confidence,
                "retrieval_confidence_assessment": confidence_assessment.as_debug_payload()
                if confidence_assessment is not None
                else None,
                "retrieval_coverage": coverage.score,
                "retrieval_intent": policy.intent.value,
                "retrieval_dominant_source": coverage.dominant_source,
                "retrieval_coverage_reason": coverage.reason,
                "partial_context": partial_context,
                "tool_calls": tool_events,
                "tool_calling_used": tool_used,
                "refused": False,
                "original_query": generation_query,
                "standalone_query": retrieval_query,
                "query_rewritten": generation_query != retrieval_query,
            }
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            logger.exception(
                "Model provider error while processing query",
                extra=self._request_log_extra(request_context),
            )
            raise map_provider_error(exc) from exc
        except Exception:
            logger.exception(
                "Failed to process query",
                extra=self._request_log_extra(request_context),
            )
            return None

    def _register_tools(self) -> None:
        for tool in self.tool_calling_config.tools:
            if tool.executor == "search_documents" and self.tool_registry.get(tool.name) is None:
                self.tool_registry.register(tool, self._execute_search_documents_tool)

    def _empty_corpus_response(self, generation_query: str, retrieval_query: str) -> dict[str, Any]:
        return {
            "result": (
                "No knowledge documents are currently indexed, so I cannot answer document-grounded questions yet. "
                "Add supported files to the documents directory and restart or reindex the application."
            ),
            "source_documents": [],
            "retrieval_confidence": 0.0,
            "retrieval_confidence_assessment": {
                "confidence": 0.0,
                "decision": RetrievalDecision.REFUSE.value,
                "reason_codes": [RetrievalReason.NO_EVIDENCE.value],
                "signals": {"coverage_reason": "empty_corpus", "candidate_count": 0},
            },
            "retrieval_coverage": 0.0,
            "retrieval_intent": None,
            "retrieval_dominant_source": None,
            "retrieval_coverage_reason": "empty_corpus",
            "partial_context": False,
            "tool_calls": [],
            "tool_calling_used": False,
            "refused": True,
            "empty_corpus": True,
            "original_query": generation_query,
            "standalone_query": retrieval_query,
            "query_rewritten": generation_query != retrieval_query,
        }

    def _refusal_response(
        self,
        generation_query: str,
        retrieval_query: str,
        policy: RetrievalPolicy,
        assessment: RetrievalConfidenceAssessment | None,
        coverage: CoverageReport,
    ) -> dict[str, Any]:
        return {
            "result": (
                "I do not have enough reliable context to answer that safely. "
                "Try rephrasing the question or ask about a more specific document, source, or section."
            ),
            "source_documents": [],
            "retrieval_confidence": assessment.score if assessment is not None else 0.0,
            "retrieval_confidence_assessment": assessment.as_debug_payload() if assessment is not None else None,
            "retrieval_coverage": coverage.score,
            "retrieval_intent": policy.intent.value,
            "retrieval_dominant_source": coverage.dominant_source,
            "retrieval_coverage_reason": coverage.reason,
            "refused": True,
            "original_query": generation_query,
            "standalone_query": retrieval_query,
            "query_rewritten": generation_query != retrieval_query,
        }

    def _reassess_documents(
        self,
        query: str,
        documents: list[Any],
        policy: RetrievalPolicy,
    ) -> tuple[list[Any], CoverageReport, RetrievalConfidenceAssessment | None]:
        if self.index_manager.context_assembler is None:
            raise RuntimeError("RAG context assembler is not initialized.")
        expanded, coverage = self.index_manager.context_assembler.expand_context(
            query,
            documents,
            policy,
        )
        return self._select_and_assess_final_context(query, expanded, policy)

    def _select_and_assess_final_context(
        self,
        query: str,
        documents: list[Any],
        policy: RetrievalPolicy,
    ) -> tuple[list[Any], CoverageReport, RetrievalConfidenceAssessment | None]:
        selected = self.context_builder.select_documents(documents)
        assembler = self.index_manager.context_assembler
        if assembler is None:
            raise RuntimeError("RAG context assembler is not initialized.")
        coverage = assembler.evaluate_coverage(selected, policy)
        confidence = self.index_manager.confidence_estimator.assess(selected, coverage, policy)
        return selected, coverage, confidence

    @staticmethod
    def _is_recoverable_assessment(assessment: RetrievalConfidenceAssessment) -> bool:
        return (
            assessment.score >= 0.20
            and RetrievalReason.NO_EVIDENCE.value not in assessment.reason_codes
        )

    def _should_refuse_retrieval(self, retrieval_result: RetrievalResult) -> bool:
        assessment = retrieval_result.confidence
        if assessment is None:
            return retrieval_result.confidence_score < self.settings.retrieval_confidence_threshold
        if assessment.decision == RetrievalDecision.ANSWER:
            return False
        return not self._is_partial_enumeration_allowed(retrieval_result)

    def _is_partial_enumeration_allowed(self, retrieval_result: RetrievalResult) -> bool:
        assessment = retrieval_result.confidence
        return bool(
            assessment
            and assessment.decision == RetrievalDecision.ANSWER
            and RetrievalReason.PARTIAL_ENUMERATION_EVIDENCE.value in assessment.reason_codes
        )

    def _build_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self.tool_registry.available()
        ]

    def _execute_search_documents_tool(
        self,
        tool_definition: ToolDefinition,
        arguments: Mapping[str, Any],
        *,
        request_context: RequestContext | None = None,
    ) -> tuple[str, list[Any], dict[str, Any]]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            payload = {"error": "Missing required argument 'query'."}
            return json.dumps(payload, ensure_ascii=True), [], payload

        maximum_top_k = self._tool_integer_property(tool_definition, "top_k", "maximum", fallback=6)
        default_top_k = self._tool_integer_property(tool_definition, "top_k", "default", fallback=maximum_top_k)
        raw_top_k = arguments.get("top_k", default_top_k)
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = default_top_k

        top_k = max(1, min(top_k, maximum_top_k))
        policy = self.policy_resolver.resolve(query)
        tool_policy = RetrievalPolicy(
            intent=policy.intent,
            retrieval_limit=max(policy.retrieval_limit, top_k * 2),
            context_limit=max(policy.context_limit, top_k),
            expand_by_source=policy.expand_by_source,
            max_sources=policy.max_sources,
            source_chunk_limit=max(policy.source_chunk_limit, top_k),
            coverage_min_chunks=policy.coverage_min_chunks,
            coverage_min_ratio=policy.coverage_min_ratio,
            retry_retrieval_limit=max(policy.retry_retrieval_limit, top_k * 2),
            retry_context_limit=max(policy.retry_context_limit, top_k),
            dense_k=max(policy.dense_k, top_k),
            dense_fetch_k=max(policy.dense_fetch_k, top_k * 2),
            sparse_k=max(policy.sparse_k, top_k),
            partial_when_incomplete=policy.partial_when_incomplete,
            scope=policy.scope,
        )
        if self.index_manager.retrieval_service is None:
            payload = {"error": "Retriever is not initialized."}
            return json.dumps(payload, ensure_ascii=True), [], payload
        retrieval_result = self.index_manager.retrieval_service.retrieve(
            query,
            tool_policy,
            request_context=request_context,
        )
        documents = retrieval_result.documents
        coverage = retrieval_result.coverage
        confidence_assessment = retrieval_result.confidence
        limited_documents = list(documents[:top_k])
        payload = {
            "query": query,
            "retrieval_confidence": retrieval_result.confidence_score,
            "retrieval_confidence_assessment": confidence_assessment.as_debug_payload()
            if confidence_assessment is not None
            else None,
            "coverage_score": coverage.score if coverage is not None else 0.0,
            "coverage_reason": coverage.reason if coverage is not None else None,
            "dominant_source": coverage.dominant_source if coverage is not None else None,
            "results": [
                {
                    "source": document.metadata.get("source", "unknown"),
                    "header_path": document.metadata.get("header_path", ""),
                    "snippet": document.page_content.strip(),
                }
                for document in limited_documents
            ],
        }
        return json.dumps(payload, ensure_ascii=True), limited_documents, payload

    def _generate_answer_with_optional_tools(
        self,
        query: str,
        documents: tuple[Any, ...],
        policy: RetrievalPolicy,
        *,
        standalone_query: str | None = None,
        coverage: CoverageReport | None = None,
        confidence_assessment: RetrievalConfidenceAssessment | None = None,
        partial_context: bool = False,
        recovery_only: bool = False,
        request_context: RequestContext | None = None,
    ) -> tuple[str, list[Any], list[dict[str, Any]], bool]:
        if recovery_only and not self.settings.tool_calling_enabled:
            return "", list(documents), [], False
        if not self.settings.tool_calling_enabled:
            return (
                self.answer_service.generate(query, list(documents), policy, coverage=coverage, partial_context=partial_context),
                list(documents),
                [],
                False,
            )

        if (
            confidence_assessment is not None
            and confidence_assessment.decision == RetrievalDecision.ANSWER
            and not partial_context
        ):
            return (
                self.answer_service.generate(query, list(documents), policy, coverage=coverage, partial_context=partial_context),
                list(documents),
                [],
                False,
            )

        user_content = query
        if standalone_query and standalone_query != query:
            user_content = (
                f"Original user question: {query}\n"
                f"Standalone retrieval query: {standalone_query}\n"
                "Answer the original user question naturally, using only the retrieved document context and any tool results."
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.answer_service.build_chat_system_prompt(
                    standalone_query or query,
                    list(documents),
                    policy,
                    coverage=coverage,
                    partial_context=partial_context,
                ),
            },
            {"role": "user", "content": user_content},
        ]
        tool_specs = self._build_tool_specs()
        collected_documents = list(documents)
        tool_events: list[dict[str, Any]] = []
        tool_used = False
        executed_search_queries: set[str] = set()
        final_tool_answer: str | None = None

        try:
            rounds = 1 if recovery_only else self.settings.tool_calling_max_rounds + 1
            for _ in range(rounds):
                with metrics.timer("tool_decision_seconds"):
                    response = self.chat_model.chat(messages, tools=tool_specs, model=self.settings.tool_calling_model)
                self._append_assistant_message(messages, content=response.content, tool_calls=response.tool_calls)

                if not response.tool_calls:
                    if tool_used and response.content.strip():
                        final_tool_answer = response.content.strip()
                    break

                tool_used = True
                for tool_call in response.tool_calls:
                    if tool_call.name == "search_documents":
                        query_arg = str(tool_call.arguments.get("query") or "").strip().casefold()
                        if query_arg in executed_search_queries:
                            tool_event = {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result_count": 0,
                                "payload": {"skipped": True, "reason": "duplicate_tool_query"},
                            }
                            tool_events.append(tool_event)
                            messages.append(
                                {
                                    "role": "tool",
                                    "name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(tool_event["payload"], ensure_ascii=True),
                                }
                            )
                            continue
                        executed_search_queries.add(query_arg)

                    with metrics.timer("tool_execution_seconds", tool=tool_call.name):
                        tool_content, tool_documents, tool_payload = self.tool_registry.execute(
                            tool_call.name,
                            tool_call.arguments,
                            request_context=request_context,
                        )
                    collected_documents = self._merge_documents(collected_documents, tool_documents)
                    tool_event = {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result_count": len(tool_documents),
                        "payload": tool_payload,
                    }
                    tool_events.append(tool_event)
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "content": tool_content,
                        }
                    )
        except Exception:
            logger.warning(
                "Tool-calling flow failed; falling back to plain generation",
                extra=self._request_log_extra(request_context),
            )

        if recovery_only:
            return "", collected_documents, tool_events, tool_used

        if final_tool_answer is not None:
            return (
                self.answer_service.validate_citations(final_tool_answer, collected_documents),
                collected_documents,
                tool_events,
                tool_used,
            )

        fallback_answer = self.answer_service.generate(
            query,
            collected_documents,
            policy,
            coverage=coverage,
            partial_context=partial_context,
        )
        return fallback_answer, collected_documents, tool_events, tool_used

    async def _agenerate_answer_with_optional_tools(
        self,
        query: str,
        documents: tuple[Any, ...],
        policy: RetrievalPolicy,
        *,
        standalone_query: str | None = None,
        coverage: CoverageReport | None = None,
        confidence_assessment: RetrievalConfidenceAssessment | None = None,
        partial_context: bool = False,
        recovery_only: bool = False,
        request_context: RequestContext | None = None,
    ) -> tuple[str, list[Any], list[dict[str, Any]], bool]:
        if recovery_only and not self.settings.tool_calling_enabled:
            return "", list(documents), [], False
        if not self.settings.tool_calling_enabled:
            return (
                await self.answer_service.agenerate(
                    query,
                    list(documents),
                    policy,
                    coverage=coverage,
                    partial_context=partial_context,
                ),
                list(documents),
                [],
                False,
            )

        if (
            confidence_assessment is not None
            and confidence_assessment.decision == RetrievalDecision.ANSWER
            and not partial_context
        ):
            return (
                await self.answer_service.agenerate(
                    query,
                    list(documents),
                    policy,
                    coverage=coverage,
                    partial_context=partial_context,
                ),
                list(documents),
                [],
                False,
            )

        user_content = query
        if standalone_query and standalone_query != query:
            user_content = (
                f"Original user question: {query}\n"
                f"Standalone retrieval query: {standalone_query}\n"
                "Answer the original user question naturally, using only the retrieved document context and any tool results."
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.answer_service.build_chat_system_prompt(
                    standalone_query or query,
                    list(documents),
                    policy,
                    coverage=coverage,
                    partial_context=partial_context,
                ),
            },
            {"role": "user", "content": user_content},
        ]
        tool_specs = self._build_tool_specs()
        collected_documents = list(documents)
        tool_events: list[dict[str, Any]] = []
        tool_used = False
        executed_search_queries: set[str] = set()
        final_tool_answer: str | None = None

        try:
            rounds = 1 if recovery_only else self.settings.tool_calling_max_rounds + 1
            for _ in range(rounds):
                with metrics.timer("tool_decision_seconds"):
                    response = await self._achat(messages, tools=tool_specs, model=self.settings.tool_calling_model)
                self._append_assistant_message(messages, content=response.content, tool_calls=response.tool_calls)

                if not response.tool_calls:
                    if tool_used and response.content.strip():
                        final_tool_answer = response.content.strip()
                    break

                tool_used = True
                for tool_call in response.tool_calls:
                    if tool_call.name == "search_documents":
                        query_arg = str(tool_call.arguments.get("query") or "").strip().casefold()
                        if query_arg in executed_search_queries:
                            tool_event = {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result_count": 0,
                                "payload": {"skipped": True, "reason": "duplicate_tool_query"},
                            }
                            tool_events.append(tool_event)
                            messages.append(
                                {
                                    "role": "tool",
                                    "name": tool_call.name,
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps(tool_event["payload"], ensure_ascii=True),
                                }
                            )
                            continue
                        executed_search_queries.add(query_arg)

                    with metrics.timer("tool_execution_seconds", tool=tool_call.name):
                        tool_content, tool_documents, tool_payload = await self._run_blocking(
                            self.tool_registry.execute,
                            tool_call.name,
                            tool_call.arguments,
                            request_context=request_context,
                        )
                    collected_documents = self._merge_documents(collected_documents, tool_documents)
                    tool_event = {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result_count": len(tool_documents),
                        "payload": tool_payload,
                    }
                    tool_events.append(tool_event)
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "content": tool_content,
                        }
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Tool-calling flow failed; falling back to plain generation",
                extra=self._request_log_extra(request_context),
            )

        if recovery_only:
            return "", collected_documents, tool_events, tool_used

        if final_tool_answer is not None:
            return (
                self.answer_service.validate_citations(final_tool_answer, collected_documents),
                collected_documents,
                tool_events,
                tool_used,
            )

        fallback_answer = await self.answer_service.agenerate(
            query,
            collected_documents,
            policy,
            coverage=coverage,
            partial_context=partial_context,
        )
        return fallback_answer, collected_documents, tool_events, tool_used

    async def _achat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> Any:
        async_chat = getattr(self.chat_model, "achat", None)
        if async_chat is not None:
            return await async_chat(messages, tools=tools, model=model)
        return await asyncio.to_thread(self.chat_model.chat, messages, tools=tools, model=model)

    async def _run_blocking(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Keep the RAG activity slot held until the worker releases its
            # backend resources; reindexing must not race a cancelled search.
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise

    def _append_assistant_message(
        self,
        messages: list[dict[str, Any]],
        *,
        content: str,
        tool_calls: list[ToolCall],
    ) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                }
                for tool_call in tool_calls
            ]
        messages.append(message)

    def _tool_integer_property(
        self,
        tool_definition: ToolDefinition,
        property_name: str,
        config_key: str,
        *,
        fallback: int,
    ) -> int:
        properties = tool_definition.parameters.get("properties")
        if not isinstance(properties, dict):
            return fallback

        property_config = properties.get(property_name)
        if not isinstance(property_config, dict):
            return fallback

        raw_value = property_config.get(config_key)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return fallback

    def _document_key(self, document: Any) -> tuple[str, str]:
        source = str(document.metadata.get("source", "unknown"))
        content = str(getattr(document, "page_content", ""))
        return source, content

    def _merge_documents(self, base_documents: list[Any], extra_documents: list[Any]) -> list[Any]:
        merged = list(base_documents)
        seen = {self._document_key(document) for document in merged}
        for document in extra_documents:
            key = self._document_key(document)
            if key in seen:
                continue
            merged.append(document)
            seen.add(key)
        return merged

    def _request_log_extra(self, request_context: RequestContext | None) -> dict[str, str]:
        if request_context is None:
            return {}
        return {
            "request_id": request_context.request_id,
            "conversation_id": request_context.conversation_id,
        }

    def interactive_chat(self) -> None:
        print("\nRAG chatbot started.")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not user_input:
                continue

            print("\nProcessing...\n")
            result = self.chat(user_input)
            if result:
                print(f"Assistant: {result['result']}\n")
                if result.get("source_documents"):
                    print("Sources:")
                    for document in result["source_documents"]:
                        print(f"  - {document.metadata.get('source', 'Unknown')}")
                    print()


def main() -> None:
    from local_rag_chatbot.application import create_application_services
    from local_rag_chatbot.settings import get_settings

    services = create_application_services(get_settings())
    chatbot = services.chatbot

    if not chatbot.load_existing_vectorstore():
        if not chatbot.setup_rag():
            print("Failed to initialize RAG.")
            return

    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
