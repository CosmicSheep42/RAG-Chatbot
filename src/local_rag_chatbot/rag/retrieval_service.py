from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any

from local_rag_chatbot.observability import metrics
from local_rag_chatbot.request_context import RequestContext
from local_rag_chatbot.rag.context_assembler import DocumentContextAssembler
from local_rag_chatbot.rag.contracts import Retriever
from local_rag_chatbot.rag.retrieval_confidence import (
    RetrievalConfidenceEstimator,
    RetrievalDecision,
)
from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicy
from local_rag_chatbot.rag.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        context_assembler: DocumentContextAssembler,
        confidence_estimator: RetrievalConfidenceEstimator | None = None,
    ) -> None:
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.confidence_estimator = confidence_estimator or RetrievalConfidenceEstimator()

    def retrieve(
        self,
        query: str,
        policy: RetrievalPolicy,
        *,
        request_context: RequestContext | None = None,
    ) -> RetrievalResult:
        metadata: dict[str, Any] = {"retried": False}
        if request_context is not None:
            metadata["request_id"] = request_context.request_id
            metadata["conversation_id"] = request_context.conversation_id
            logger.info(
                "Retrieving documents",
                extra={
                    "request_id": request_context.request_id,
                    "conversation_id": request_context.conversation_id,
                },
            )
        initial_documents = self._retrieve_candidates(query, policy)
        with metrics.timer("context_assembly_seconds"):
            expanded_documents, coverage = self.context_assembler.expand_context(
                query,
                initial_documents,
                policy,
            )
        with metrics.timer("retrieval_confidence_seconds"):
            confidence = self.confidence_estimator.assess(expanded_documents, coverage, policy)

        final_policy = policy
        final_candidates = initial_documents
        if confidence.decision == RetrievalDecision.RETRY:
            metrics.increment("retrieval_retries_total")
            retry_policy = self._retry_policy(policy)
            retry_documents = self._retrieve_candidates(query, retry_policy)
            with metrics.timer("context_assembly_seconds", retry=True):
                retry_expanded, retry_coverage = self.context_assembler.expand_context(
                    query,
                    retry_documents,
                    retry_policy,
                )
            with metrics.timer("retrieval_confidence_seconds", retry=True):
                retry_confidence = self.confidence_estimator.assess(
                    retry_expanded,
                    retry_coverage,
                    retry_policy,
                )
            if self._is_better_result(retry_confidence, retry_coverage, confidence, coverage):
                expanded_documents = retry_expanded
                coverage = retry_coverage
                confidence = retry_confidence
                final_candidates = retry_documents
                final_policy = retry_policy
                metadata["retried"] = True

        return RetrievalResult(
            query=query,
            documents=expanded_documents,
            candidates=final_candidates,
            confidence=confidence,
            policy=final_policy,
            coverage=coverage,
            reason_codes=confidence.reason_codes,
            metadata=metadata,
        )

    @staticmethod
    def _is_better_result(retry_confidence, retry_coverage, confidence, coverage) -> bool:
        retry_answers = retry_confidence.decision == RetrievalDecision.ANSWER
        current_answers = confidence.decision == RetrievalDecision.ANSWER
        if retry_answers != current_answers:
            return retry_answers
        return (retry_confidence.score, retry_coverage.score) > (confidence.score, coverage.score)

    def _retrieve_candidates(self, query: str, policy: RetrievalPolicy) -> list[Any]:
        return self.retriever.retrieve(
            query,
            candidate_limit=policy.retrieval_limit,
            final_limit=policy.context_limit,
            dense_k=policy.dense_k,
            dense_fetch_k=policy.dense_fetch_k,
            sparse_k=policy.sparse_k,
        )

    def _retry_policy(self, policy: RetrievalPolicy) -> RetrievalPolicy:
        return replace(
            policy,
            retrieval_limit=policy.retry_retrieval_limit,
            context_limit=policy.retry_context_limit,
            source_chunk_limit=max(policy.source_chunk_limit, policy.retry_context_limit),
            dense_k=max(policy.dense_k or policy.retrieval_limit, policy.retry_retrieval_limit),
            dense_fetch_k=max(
                policy.dense_fetch_k or policy.retry_retrieval_limit,
                policy.retry_retrieval_limit * 2,
            ),
            sparse_k=max(policy.sparse_k or policy.retrieval_limit, policy.retry_retrieval_limit),
        )
