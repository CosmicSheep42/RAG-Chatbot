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


def passage_recall_at_k(
    results: Sequence[Mapping[str, object]], passages: Iterable[Mapping[str, object]], k: int
) -> float:
    """Return the fraction of gold passages touched by retrieved source ranges.

    This intentionally measures *touch*, not sufficient evidence coverage. Use
    :func:`passage_coverage_at_k` for the stricter coverage metric.
    """
    required = list(passages)
    if not required:
        return math.nan
    retrieved = results[: max(k, 0)]
    valid_passages = [
        passage
        for passage in required
        if isinstance(passage.get("start_char"), int)
        and isinstance(passage.get("end_char"), int)
        and int(passage["end_char"]) > int(passage["start_char"])
    ]
    if not valid_passages:
        return math.nan
    valid_results = [
        result
        for result in retrieved
        if isinstance(result.get("start_char"), int) and isinstance(result.get("end_char"), int)
    ]
    if retrieved and not valid_results:
        return math.nan
    matched = 0
    for passage in valid_passages:
        source = str(passage.get("document_path", ""))
        start = passage.get("start_char")
        end = passage.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if any(
            str(result.get("document_id", "")) == source
            and isinstance(result.get("start_char"), int)
            and isinstance(result.get("end_char"), int)
            and int(result["start_char"]) < end
            and int(result["end_char"]) > start
            for result in retrieved
        ):
            matched += 1
    return matched / len(valid_passages)


def passage_coverage_at_k(
    results: Sequence[Mapping[str, object]],
    passages: Iterable[Mapping[str, object]],
    k: int,
    *,
    threshold: float = 0.8,
) -> float:
    """Return the fraction of gold passage length covered by retrieved chunks.

    Overlapping retrieved ranges are merged before measuring coverage. The
    threshold is validated so experiment metadata cannot describe an invalid
    coverage policy.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    required = [
        passage
        for passage in passages
        if isinstance(passage.get("start_char"), int)
        and isinstance(passage.get("end_char"), int)
        and int(passage["end_char"]) > int(passage["start_char"])
    ]
    if not required:
        return math.nan
    retrieved = [
        result
        for result in results[: max(k, 0)]
        if isinstance(result.get("start_char"), int) and isinstance(result.get("end_char"), int)
    ]
    if results and not retrieved:
        return math.nan
    covered = 0
    for passage in required:
        source = str(passage.get("document_path", ""))
        start = int(passage["start_char"])
        end = int(passage["end_char"])
        intervals = sorted(
            (
                max(start, int(result["start_char"])),
                min(end, int(result["end_char"])),
            )
            for result in retrieved
            if str(result.get("document_id", "")) == source
            and int(result["end_char"]) > start
            and int(result["start_char"]) < end
        )
        merged: list[list[int]] = []
        for left, right in intervals:
            if right <= left:
                continue
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        covered += sum(right - left for left, right in merged)
    total = sum(int(passage["end_char"]) - int(passage["start_char"]) for passage in required)
    return covered / total if total else math.nan


def complete_passage_retrieval(
    results: Sequence[Mapping[str, object]],
    passages: Iterable[Mapping[str, object]],
    k: int,
    *,
    threshold: float = 0.8,
) -> bool | None:
    required = [
        passage
        for passage in passages
        if isinstance(passage.get("start_char"), int)
        and isinstance(passage.get("end_char"), int)
        and int(passage["end_char"]) > int(passage["start_char"])
    ]
    if not required:
        return None
    coverages = [passage_coverage_at_k(results, [passage], k, threshold=threshold) for passage in required]
    if any(math.isnan(coverage) for coverage in coverages):
        return None
    return all(coverage >= threshold for coverage in coverages)


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
