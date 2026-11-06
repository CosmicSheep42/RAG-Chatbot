"""Deterministic metrics for saved benchmark retrieval and answer records."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _document_ids(results: Sequence[Mapping[str, object] | str]) -> list[str]:
    values = []
    for result in results:
        if isinstance(result, str):
            values.append(result)
        else:
            values.append(str(result.get("document_id") or result.get("source") or ""))
    return _unique(value for value in values if value)


def recall_at_k(results: Sequence[Mapping[str, object] | str], relevant: Iterable[str], k: int) -> float:
    required = set(_unique(relevant))
    if not required:
        return math.nan
    found = set(_document_ids(results[: max(k, 0)]))
    return len(found & required) / len(required)


def complete_evidence_retrieval(
    results: Sequence[Mapping[str, object] | str], relevant: Iterable[str], k: int
) -> bool | None:
    required = set(_unique(relevant))
    if not required:
        return None
    return required.issubset(_document_ids(results[: max(k, 0)]))


def reciprocal_rank_at_k(results: Sequence[Mapping[str, object] | str], relevant: Iterable[str], k: int) -> float:
    required = set(_unique(relevant))
    if not required:
        return math.nan
    for rank, document_id in enumerate(_document_ids(results[: max(k, 0)]), start=1):
        if document_id in required:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: Sequence[Mapping[str, object] | str], relevant: Iterable[str], k: int) -> float:
    required = set(_unique(relevant))
    if not required:
        return math.nan
    retrieved = _document_ids(results[: max(k, 0)])
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, value in enumerate(retrieved, 1) if value in required)
    ideal = sum((1.0 / math.log2(rank + 1)) for rank in range(1, min(len(required), max(k, 0)) + 1))
    return dcg / ideal if ideal else 0.0


def fact_coverage(answer: str, required_facts: Iterable[str]) -> float:
    facts = [fact for fact in required_facts if str(fact).strip()]
    if not facts:
        return math.nan
    normalized_answer = _normalize(answer)
    return sum(_normalize(fact) in normalized_answer for fact in facts) / len(facts)


def citation_quality(
    citations: Iterable[str], retrieved_documents: Iterable[str], relevant_documents: Iterable[str]
) -> dict[str, float]:
    cited = set(_unique(citations))
    retrieved = set(_unique(retrieved_documents))
    relevant = set(_unique(relevant_documents))
    return {
        "validity": len(cited & retrieved) / len(cited) if cited else 0.0,
        "relevance": len(cited & relevant) / len(cited) if cited else 0.0,
    }


def citation_support(
    citations: Iterable[Mapping[str, object] | str],
    evidence_documents: Iterable[str],
) -> float:
    """Measure source-level evidence support, not claim-level groundedness."""
    evidence = set(_unique(evidence_documents))
    cited = []
    for citation in citations:
        if isinstance(citation, str):
            cited.append(citation)
        else:
            cited.append(str(citation.get("document_id") or citation.get("source") or ""))
    cited = [value for value in _unique(cited) if value]
    return sum(value in evidence for value in cited) / len(cited) if cited else 0.0


def is_abstention(answer: str) -> bool:
    return bool(
        re.search(r"\b(?:cannot|can't|unable|do not have|don't have|not enough|insufficient)\b", answer.casefold())
    )


def abstention_result(answer: str, answerable: bool) -> str:
    abstained = is_abstention(answer)
    if answerable:
        return "incorrect_refusal" if abstained else "answered"
    return "correct_abstention" if abstained else "unsupported_answer"


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))
