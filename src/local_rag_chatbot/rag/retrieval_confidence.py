from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy


class RetrievalDecision(str, Enum):
    ANSWER = "answer"
    RETRY = "retry"
    REFUSE = "refuse"


class RetrievalReason(str, Enum):
    NO_EVIDENCE = "NO_EVIDENCE"
    WEAK_RERANK = "WEAK_RERANK"
    RERANKER_UNAVAILABLE = "RERANKER_UNAVAILABLE"
    RETRIEVERS_DISAGREE = "RETRIEVERS_DISAGREE"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    STRONG_RERANK_EVIDENCE = "STRONG_RERANK_EVIDENCE"
    STRONG_MULTI_SIGNAL_EVIDENCE = "STRONG_MULTI_SIGNAL_EVIDENCE"
    PARTIAL_ENUMERATION_EVIDENCE = "PARTIAL_ENUMERATION_EVIDENCE"
    DENSE_ONLY_WEAK = "DENSE_ONLY_WEAK"
    SPARSE_ONLY_WEAK = "SPARSE_ONLY_WEAK"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    EXACT_LEXICAL_MATCH = "EXACT_LEXICAL_MATCH"


@dataclass(frozen=True)
class CandidateRetrievalEvidence:
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_rrf_score: float | None = None
    reranker_score: float | None = None
    metadata_boost: float = 0.0
    final_ranking_score: float | None = None
    exact_lexical_match: bool = False

    @classmethod
    def from_document(cls, document: Any) -> CandidateRetrievalEvidence:
        metadata = getattr(document, "metadata", {}) or {}
        reranker_score = _float_metadata(metadata, "retrieval_reranker_score")
        if reranker_score is None:
            reranker_score = _float_metadata(metadata, "retrieval_rerank_score")
        return cls(
            dense_score=_float_metadata(metadata, "retrieval_dense_score"),
            dense_rank=_int_metadata(metadata, "retrieval_dense_rank"),
            sparse_score=_float_metadata(metadata, "retrieval_sparse_score"),
            sparse_rank=_int_metadata(metadata, "retrieval_sparse_rank"),
            fused_rrf_score=_float_metadata(metadata, "retrieval_fused_rrf_score"),
            reranker_score=reranker_score,
            metadata_boost=_float_metadata(metadata, "retrieval_metadata_boost") or 0.0,
            final_ranking_score=_float_metadata(metadata, "retrieval_final_ranking_score")
            or _float_metadata(metadata, "retrieval_score"),
            exact_lexical_match=bool(metadata.get("retrieval_exact_match") or metadata.get("retrieval_exact_code_match")),
        )


@dataclass(frozen=True)
class RetrievalConfidenceProfile:
    answer_threshold: float
    retry_threshold: float
    strong_reranker_threshold: float
    weak_reranker_threshold: float
    min_coverage_score: float
    low_margin_threshold: float = 0.08


@dataclass(frozen=True)
class RetrievalConfidenceConfig:
    default_profile: RetrievalConfidenceProfile = RetrievalConfidenceProfile(
        answer_threshold=0.62,
        retry_threshold=0.35,
        strong_reranker_threshold=0.70,
        weak_reranker_threshold=0.25,
        min_coverage_score=0.20,
    )
    intent_profiles: dict[QueryIntent, RetrievalConfidenceProfile] = field(default_factory=dict)

    @classmethod
    def default(cls) -> RetrievalConfidenceConfig:
        return cls(
            intent_profiles={
                QueryIntent.ENUMERATION: RetrievalConfidenceProfile(0.66, 0.40, 0.72, 0.28, 0.55),
                QueryIntent.AGGREGATION: RetrievalConfidenceProfile(0.66, 0.42, 0.72, 0.30, 0.55),
                QueryIntent.SUMMARY: RetrievalConfidenceProfile(0.58, 0.34, 0.66, 0.22, 0.25),
                QueryIntent.BROAD_SEMANTIC: RetrievalConfidenceProfile(0.60, 0.35, 0.68, 0.24, 0.25),
                QueryIntent.COMPARISON: RetrievalConfidenceProfile(0.64, 0.40, 0.70, 0.28, 0.45),
                QueryIntent.METADATA_LOOKUP: RetrievalConfidenceProfile(0.60, 0.34, 0.68, 0.24, 0.20),
            }
        )

    def profile_for(self, intent: QueryIntent) -> RetrievalConfidenceProfile:
        return self.intent_profiles.get(intent, self.default_profile)


@dataclass(frozen=True)
class RetrievalConfidenceAssessment:
    confidence: float
    decision: RetrievalDecision
    reason_codes: tuple[str, ...]
    signals: dict[str, Any]

    @property
    def score(self) -> float:
        return self.confidence

    def as_debug_payload(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 6),
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "signals": self.signals,
        }


class RetrievalConfidenceEstimator:
    def __init__(self, config: RetrievalConfidenceConfig | None = None) -> None:
        self.config = config or RetrievalConfidenceConfig.default()

    def assess(
        self,
        candidates: list[Any],
        coverage: Any,
        policy: RetrievalPolicy,
    ) -> RetrievalConfidenceAssessment:
        evidence_documents = _retrieval_evidence_documents(candidates)
        evidence = [CandidateRetrievalEvidence.from_document(candidate) for candidate in evidence_documents]
        if not evidence:
            return self._assessment(
                0.0,
                RetrievalDecision.REFUSE,
                [RetrievalReason.NO_EVIDENCE],
                evidence,
                coverage,
                policy,
            )

        profile = self.config.profile_for(policy.intent)
        top = evidence[0]
        second = evidence[1] if len(evidence) > 1 else None
        reranker_evidence = _reranker_evidence(top.reranker_score)
        lexical_evidence = _lexical_evidence(top.sparse_score)
        dense_evidence = _dense_evidence(top)
        agreement = _retriever_agreement(top)
        exact_match = 1.0 if top.exact_lexical_match else 0.0
        independent_signals = _independent_signal_count(top, reranker_evidence, lexical_evidence, dense_evidence)
        margin = _reranker_margin(top, second)

        confidence = self._combine_evidence(
            reranker_evidence=reranker_evidence,
            lexical_evidence=lexical_evidence,
            dense_evidence=dense_evidence,
            agreement=agreement,
            exact_match=exact_match,
            independent_signals=independent_signals,
        )

        reason_codes: list[RetrievalReason] = []
        if reranker_evidence is None:
            reason_codes.append(RetrievalReason.RERANKER_UNAVAILABLE)
        elif reranker_evidence < profile.weak_reranker_threshold and lexical_evidence < 0.35:
            reason_codes.append(RetrievalReason.WEAK_RERANK)

        if exact_match:
            reason_codes.append(RetrievalReason.EXACT_LEXICAL_MATCH)
        if lexical_evidence <= 0 and dense_evidence > 0:
            reason_codes.append(RetrievalReason.DENSE_ONLY_WEAK)
        if lexical_evidence > 0 and dense_evidence <= 0 and reranker_evidence is None:
            reason_codes.append(RetrievalReason.SPARSE_ONLY_WEAK)
        if dense_evidence > 0 and lexical_evidence <= 0 and (reranker_evidence is None or reranker_evidence < profile.strong_reranker_threshold):
            reason_codes.append(RetrievalReason.RETRIEVERS_DISAGREE)
        if margin is not None and margin < profile.low_margin_threshold and confidence >= profile.retry_threshold:
            reason_codes.append(RetrievalReason.AMBIGUOUS_EVIDENCE)

        coverage_sufficient = bool(getattr(coverage, "sufficient", False))
        coverage_score = _float_value(getattr(coverage, "score", None)) or 0.0
        coverage_required = _coverage_required(policy)
        if coverage_required and (not coverage_sufficient or coverage_score < profile.min_coverage_score):
            reason_codes.append(RetrievalReason.INSUFFICIENT_COVERAGE)

        if independent_signals >= 2 and confidence >= profile.answer_threshold:
            reason_codes.append(RetrievalReason.STRONG_MULTI_SIGNAL_EVIDENCE)
        elif reranker_evidence is not None and reranker_evidence >= profile.strong_reranker_threshold:
            reason_codes.append(RetrievalReason.STRONG_RERANK_EVIDENCE)

        decision = self._decision(
            confidence=confidence,
            profile=profile,
            policy=policy,
            coverage_required=coverage_required,
            coverage_sufficient=coverage_sufficient,
            coverage_score=coverage_score,
            reason_codes=reason_codes,
        )
        return self._assessment(confidence, decision, reason_codes, evidence, coverage, policy)

    def _combine_evidence(
        self,
        *,
        reranker_evidence: float | None,
        lexical_evidence: float,
        dense_evidence: float,
        agreement: float,
        exact_match: float,
        independent_signals: int,
    ) -> float:
        scores = [0.0]
        if reranker_evidence is not None:
            scores.append(reranker_evidence)
        if lexical_evidence > 0:
            scores.append(min(0.62, 0.12 + 0.50 * lexical_evidence))
        if lexical_evidence > 0 and dense_evidence > 0:
            scores.append(min(0.86, 0.18 + 0.38 * lexical_evidence + 0.22 * dense_evidence + agreement))
        if exact_match:
            scores.append(min(0.88, 0.45 + 0.28 * lexical_evidence + 0.15 * dense_evidence))

        confidence = max(scores)
        if independent_signals >= 2:
            confidence = min(1.0, confidence + 0.08)
        if exact_match:
            confidence = min(1.0, confidence + 0.06)
        return max(0.0, min(1.0, confidence))

    def _decision(
        self,
        *,
        confidence: float,
        profile: RetrievalConfidenceProfile,
        policy: RetrievalPolicy,
        coverage_required: bool,
        coverage_sufficient: bool,
        coverage_score: float,
        reason_codes: list[RetrievalReason],
    ) -> RetrievalDecision:
        has_insufficient_coverage = RetrievalReason.INSUFFICIENT_COVERAGE in reason_codes
        has_strong_signal = (
            RetrievalReason.STRONG_MULTI_SIGNAL_EVIDENCE in reason_codes
            or RetrievalReason.STRONG_RERANK_EVIDENCE in reason_codes
            or RetrievalReason.EXACT_LEXICAL_MATCH in reason_codes
        )
        has_weak_rerank = RetrievalReason.WEAK_RERANK in reason_codes
        has_dense_only = RetrievalReason.DENSE_ONLY_WEAK in reason_codes

        if confidence >= profile.answer_threshold and has_strong_signal:
            if has_insufficient_coverage:
                if policy.partial_when_incomplete and policy.intent in {QueryIntent.ENUMERATION, QueryIntent.AGGREGATION}:
                    reason_codes.append(RetrievalReason.PARTIAL_ENUMERATION_EVIDENCE)
                    return RetrievalDecision.ANSWER
                return RetrievalDecision.RETRY
            return RetrievalDecision.ANSWER

        if confidence < profile.retry_threshold:
            return RetrievalDecision.REFUSE

        if has_weak_rerank and not has_strong_signal:
            return RetrievalDecision.REFUSE

        if has_dense_only and not has_strong_signal:
            return RetrievalDecision.REFUSE

        if coverage_required and (not coverage_sufficient or coverage_score < profile.min_coverage_score):
            return RetrievalDecision.RETRY

        return RetrievalDecision.RETRY

    def _assessment(
        self,
        confidence: float,
        decision: RetrievalDecision,
        reason_codes: list[RetrievalReason],
        evidence: list[CandidateRetrievalEvidence],
        coverage: Any,
        policy: RetrievalPolicy,
    ) -> RetrievalConfidenceAssessment:
        top = evidence[0] if evidence else CandidateRetrievalEvidence()
        unique_reasons = tuple(dict.fromkeys(reason.value for reason in reason_codes))
        if not unique_reasons:
            unique_reasons = (RetrievalReason.AMBIGUOUS_EVIDENCE.value,)
        signals = {
            "top_candidate": {
                "dense_score": top.dense_score,
                "dense_rank": top.dense_rank,
                "sparse_score": top.sparse_score,
                "sparse_rank": top.sparse_rank,
                "reranker_score": top.reranker_score,
                "metadata_boost": round(top.metadata_boost, 6),
                "final_ranking_score": top.final_ranking_score,
                "exact_lexical_match": top.exact_lexical_match,
            },
            "reranker_evidence": _reranker_evidence(top.reranker_score),
            "lexical_evidence": _lexical_evidence(top.sparse_score),
            "dense_evidence": _dense_evidence(top),
            "retriever_agreement": _retriever_agreement(top),
            "independent_evidence_count": _independent_signal_count(
                top,
                _reranker_evidence(top.reranker_score),
                _lexical_evidence(top.sparse_score),
                _dense_evidence(top),
            ),
            "reranker_margin": _reranker_margin(top, evidence[1] if len(evidence) > 1 else None),
            "coverage_score": round(_float_value(getattr(coverage, "score", None)) or 0.0, 6),
            "coverage_sufficient": bool(getattr(coverage, "sufficient", False)),
            "coverage_reason": getattr(coverage, "reason", ""),
            "intent": policy.intent.value,
            "candidate_count": len(evidence),
        }
        return RetrievalConfidenceAssessment(
            confidence=max(0.0, min(1.0, confidence)),
            decision=decision,
            reason_codes=unique_reasons,
            signals=signals,
        )


def _coverage_required(policy: RetrievalPolicy) -> bool:
    return policy.expand_by_source or policy.intent in {
        QueryIntent.ENUMERATION,
        QueryIntent.AGGREGATION,
        QueryIntent.COMPARISON,
        QueryIntent.SUMMARY,
        QueryIntent.BROAD_SEMANTIC,
    }


def _retrieval_evidence_documents(candidates: list[Any]) -> list[Any]:
    hits = [
        candidate
        for candidate in candidates
        if (getattr(candidate, "metadata", {}) or {}).get("context_role") == "hit"
    ]
    if hits:
        return hits
    return [
        candidate
        for candidate in candidates
        if _has_retrieval_metadata(getattr(candidate, "metadata", {}) or {})
    ]


def _has_retrieval_metadata(metadata: dict[str, Any]) -> bool:
    return any(
        key in metadata
        for key in (
            "retrieval_score",
            "retrieval_rank",
            "retrieval_dense_score",
            "retrieval_sparse_score",
            "retrieval_reranker_score",
            "retrieval_rerank_score",
            "retrieval_exact_match",
            "retrieval_exact_code_match",
            "retrieval_final_ranking_score",
        )
    )


def _float_metadata(metadata: dict[str, Any], key: str) -> float | None:
    return _float_value(metadata.get(key))


def _int_metadata(metadata: dict[str, Any], key: str) -> int | None:
    value = _float_value(metadata.get(key))
    if value is None:
        return None
    return int(value)


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _lexical_evidence(sparse_score: float | None) -> float:
    if sparse_score is None or sparse_score <= 0:
        return 0.0
    return max(0.0, min(1.0, math.log1p(sparse_score) / 2.0))


def _dense_evidence(evidence: CandidateRetrievalEvidence) -> float:
    if evidence.dense_score is not None and evidence.dense_score > 0:
        return max(0.0, min(0.65, evidence.dense_score))
    return 0.0


def _reranker_evidence(reranker_score: float | None) -> float | None:
    if reranker_score is None:
        return None
    try:
        return 1.0 / (1.0 + math.exp(-reranker_score))
    except OverflowError:
        return 0.0 if reranker_score < 0 else 1.0


def _retriever_agreement(evidence: CandidateRetrievalEvidence) -> float:
    if evidence.dense_rank is None or evidence.sparse_rank is None:
        return 0.0
    proximity = 1.0 / max(abs(evidence.dense_rank - evidence.sparse_rank) + 1, 1)
    return min(0.12, 0.08 + 0.04 * proximity)


def _independent_signal_count(
    evidence: CandidateRetrievalEvidence,
    reranker_evidence: float | None,
    lexical_evidence: float,
    dense_evidence: float,
) -> int:
    return sum(
        [
            1 if reranker_evidence is not None and reranker_evidence >= 0.55 else 0,
            1 if lexical_evidence > 0 else 0,
            1 if dense_evidence > 0 else 0,
            1 if evidence.exact_lexical_match else 0,
        ]
    )


def _reranker_margin(
    top: CandidateRetrievalEvidence,
    second: CandidateRetrievalEvidence | None,
) -> float | None:
    if top.reranker_score is None or second is None or second.reranker_score is None:
        return None
    return top.reranker_score - second.reranker_score
