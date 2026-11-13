from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable

from langchain_core.documents import Document

from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy

RETRIEVAL_METADATA_KEYS = {
    "retrieval_score",
    "retrieval_rank",
    "retrieval_dense_score",
    "retrieval_sparse_score",
    "retrieval_reranker_score",
    "retrieval_rerank_score",
    "retrieval_exact_match",
    "retrieval_exact_code_match",
    "retrieval_final_ranking_score",
}


BULLET_OR_NUMBERED_LIST_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[\.)])\s+\S", re.MULTILINE)
MARKDOWN_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def clone_document(document: Document) -> Document:
    return Document(page_content=document.page_content, metadata=dict(document.metadata))


@dataclass(frozen=True)
class SourceCoverage:
    source: str
    selected_chunks: int
    total_chunks: int
    selected_units: int
    total_units: int
    continuity_ratio: float
    structural_ratio: float


@dataclass(frozen=True)
class CoverageReport:
    """Structural context-distribution diagnostics, not semantic query coverage.

    ``score`` is a corpus-relative structural index; only ``sufficient`` is the
    intent-aware structural gate consumed by retrieval decisions.
    """

    sufficient: bool
    score: float
    dominant_source: str | None
    selected_sources: list[str]
    dominant_source_chunks: int
    selected_chunks: int
    total_available_chunks: int
    selected_units: int
    total_available_units: int
    continuity_ratio: float
    structural_ratio: float
    source_concentration: float
    source_reports: list[SourceCoverage] = field(default_factory=list)
    reason: str = ""

    @property
    def item_signals(self) -> int:
        return self.selected_units

    @property
    def structure_signals(self) -> int:
        return int(round(self.structural_ratio * max(self.selected_chunks, 1)))


@dataclass(frozen=True)
class _StructureKey:
    source: str
    path: str


class DocumentContextAssembler:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        self._documents_by_source = self._group_by_source(self.documents)
        self._source_unit_counts = {
            source: len({self._unit_key(document).path for document in source_docs})
            for source, source_docs in self._documents_by_source.items()
        }

    def expand_context(
        self,
        query: str,
        candidates: list[Document],
        policy: RetrievalPolicy,
    ) -> tuple[list[Document], CoverageReport]:
        if not candidates:
            coverage = self._evaluate_coverage([], policy)
            return [], coverage

        if not policy.expand_by_source:
            ordered = self._mark_hits(self._dedupe_by_retrieval_priority(candidates))
            selected = ordered[: policy.context_limit]
            coverage = self._evaluate_coverage(selected, policy)
            return selected, coverage

        selected_sources = self._rank_sources(query, candidates)[: policy.max_sources]
        if not selected_sources:
            ordered = self._mark_hits(self._dedupe_by_retrieval_priority(candidates))
            selected = ordered[: policy.context_limit]
            coverage = self._evaluate_coverage(selected, policy)
            return selected, coverage

        per_source_budget = self._source_budget(policy, len(selected_sources))
        retrieved_by_key = self._retrieved_by_key(candidates)
        expanded_by_source: dict[str, list[Document]] = {}
        for source in selected_sources:
            source_candidates = [document for document in candidates if self._source_key(document) == source]
            expanded = self._expand_source(
                query=query,
                source=source,
                candidates=source_candidates,
                policy=policy,
                budget=per_source_budget,
            )
            expanded_by_source[source] = self._context_documents(expanded, retrieved_by_key, source_candidates)

        interleaved = self._interleave_sources(expanded_by_source, selected_sources)
        selected = self._dedupe_preserving_order(interleaved)[: policy.context_limit]
        coverage = self._evaluate_coverage(selected, policy)
        return selected, coverage

    def evaluate_coverage(self, documents: list[Document], policy: RetrievalPolicy) -> CoverageReport:
        return self._evaluate_coverage(documents, policy)

    def expand_with_retry(
        self,
        query: str,
        retriever_docs: list[Document],
        policy: RetrievalPolicy,
    ) -> tuple[list[Document], CoverageReport]:
        expanded, coverage = self.expand_context(query, retriever_docs, policy)
        if coverage.sufficient or not policy.expand_by_source:
            return expanded, coverage

        retry_policy = RetrievalPolicy(
            intent=policy.intent,
            retrieval_limit=policy.retry_retrieval_limit,
            context_limit=policy.retry_context_limit,
            expand_by_source=policy.expand_by_source,
            max_sources=policy.max_sources,
            source_chunk_limit=max(policy.source_chunk_limit, policy.retry_context_limit),
            coverage_min_chunks=policy.coverage_min_chunks,
            coverage_min_ratio=policy.coverage_min_ratio,
            retry_retrieval_limit=policy.retry_retrieval_limit,
            retry_context_limit=policy.retry_context_limit,
            dense_k=policy.dense_k,
            dense_fetch_k=policy.dense_fetch_k,
            sparse_k=policy.sparse_k,
            partial_when_incomplete=policy.partial_when_incomplete,
            scope=policy.scope,
        )
        expanded_retry, coverage_retry = self.expand_context(query, retriever_docs, retry_policy)
        if coverage_retry.score >= coverage.score:
            return expanded_retry, coverage_retry
        return expanded, coverage

    def _expand_source(
        self,
        *,
        query: str,
        source: str,
        candidates: list[Document],
        policy: RetrievalPolicy,
        budget: int,
    ) -> list[Document]:
        source_docs = self._documents_by_source.get(source, [])
        if not source_docs:
            return self._mark_hits(self._dedupe_by_retrieval_priority(candidates)[:budget])

        candidate_units = {self._unit_key(document).path for document in candidates}
        selected: list[Document] = []

        if self._is_broad_intent(policy.intent):
            selected = self._expand_broad_source(source_docs, candidates, min(budget, policy.source_chunk_limit))
        elif policy.intent in {QueryIntent.ENUMERATION, QueryIntent.AGGREGATION}:
            selected = self._expand_structural_units(source_docs, candidates, candidate_units, budget, radius=2)
            if not selected:
                candidate_keys = {self._document_key(document) for document in candidates}
                selected = self._expand_around_candidates(source_docs, candidate_keys, budget, radius=2)
        elif policy.intent == QueryIntent.COMPARISON:
            selected = self._expand_structural_units(source_docs, candidates, candidate_units, budget, radius=1)
            if len(selected) < min(budget, 2):
                candidate_keys = {self._document_key(document) for document in candidates}
                selected = self._expand_around_candidates(source_docs, candidate_keys, budget, radius=1)
        else:
            selected = self._expand_around_relevant_candidates(source_docs, candidates, budget, radius=1)

        if self._mentions_source_scope(query, source):
            selected = self._prefer_contiguous(selected, source_docs, budget)

        return self._dedupe_preserving_order(selected)[:budget]

    def _expand_broad_source(
        self,
        source_docs: list[Document],
        candidates: list[Document],
        budget: int,
    ) -> list[Document]:
        positions_by_key = {self._document_key(document): index for index, document in enumerate(source_docs)}
        ranked_hits = [
            document for document in sorted(candidates, key=self._retrieval_priority_key)
            if self._document_key(document) in positions_by_key
        ]
        selected: list[Document] = []
        selected_keys: set[str] = set()

        for hit in ranked_hits:
            if len(selected) >= budget:
                return selected
            key = self._document_key(hit)
            if key in selected_keys:
                continue
            selected.append(source_docs[positions_by_key[key]])
            selected_keys.add(key)

        if len(selected) >= budget:
            return selected

        for document in self._broad_summary_fill_order(source_docs):
            key = self._document_key(document)
            if key in selected_keys:
                continue
            selected.append(document)
            selected_keys.add(key)
            if len(selected) >= budget:
                return selected
        return selected

    def _broad_summary_fill_order(self, source_docs: list[Document]) -> list[Document]:
        if len(source_docs) <= 2:
            return list(source_docs)
        positions = [0, len(source_docs) // 2, len(source_docs) - 1]
        ordered_positions = list(dict.fromkeys(positions))
        ordered = [source_docs[position] for position in ordered_positions]
        ordered.extend(source_docs)
        return ordered

    def _expand_structural_units(
        self,
        source_docs: list[Document],
        candidates: list[Document],
        candidate_units: set[str],
        budget: int,
        *,
        radius: int,
    ) -> list[Document]:
        if not candidate_units:
            return []

        positions_by_key = {self._document_key(document): index for index, document in enumerate(source_docs)}
        ranked_hits = [
            document for document in sorted(candidates, key=self._retrieval_priority_key)
            if self._document_key(document) in positions_by_key
        ]
        selected: list[Document] = []
        selected_keys: set[str] = set()

        for hit in ranked_hits:
            if len(selected) >= budget:
                return selected
            hit_key = self._document_key(hit)
            if hit_key in selected_keys:
                continue
            selected.append(source_docs[positions_by_key[hit_key]])
            selected_keys.add(hit_key)

        for hit in ranked_hits:
            if len(selected) >= budget:
                return selected
            hit_position = positions_by_key[self._document_key(hit)]
            for position in self._neighbor_positions(hit_position, len(source_docs), radius):
                document = source_docs[position]
                key = self._document_key(document)
                if key in selected_keys:
                    continue
                selected.append(document)
                selected_keys.add(key)
                if len(selected) >= budget:
                    return selected

        sibling_paths = self._sibling_unit_paths(source_docs, candidate_units)
        wanted_paths = list(dict.fromkeys([*candidate_units, *sibling_paths]))
        for path in wanted_paths:
            for document in source_docs:
                if self._unit_key(document).path == path:
                    key = self._document_key(document)
                    if key in selected_keys:
                        continue
                    selected.append(document)
                    selected_keys.add(key)
                    if len(selected) >= budget:
                        return selected
        return selected

    def _expand_around_candidates(
        self,
        source_docs: list[Document],
        candidate_keys: set[str],
        budget: int,
        *,
        radius: int,
    ) -> list[Document]:
        selected_positions: set[int] = set()
        for index, document in enumerate(source_docs):
            if self._document_key(document) not in candidate_keys:
                continue
            start = max(0, index - radius)
            end = min(len(source_docs), index + radius + 1)
            selected_positions.update(range(start, end))

        if not selected_positions:
            selected_positions.update(range(min(len(source_docs), budget)))

        ordered_positions = sorted(selected_positions)
        if len(ordered_positions) > budget:
            candidate_positions = [
                index for index, document in enumerate(source_docs) if self._document_key(document) in candidate_keys
            ]
            ordered_positions = self._trim_positions_around_hits(ordered_positions, candidate_positions, budget)
        return [source_docs[index] for index in ordered_positions[:budget]]

    def _expand_around_relevant_candidates(
        self,
        source_docs: list[Document],
        candidates: list[Document],
        budget: int,
        *,
        radius: int,
    ) -> list[Document]:
        if budget <= 0:
            return []

        positions_by_key = {self._document_key(document): index for index, document in enumerate(source_docs)}
        ranked_hits = [
            document for document in sorted(candidates, key=self._retrieval_priority_key)
            if self._document_key(document) in positions_by_key
        ]
        if not ranked_hits:
            return self._expand_around_candidates(source_docs, set(), budget, radius=radius)

        selected_positions: set[int] = set()
        for hit in ranked_hits:
            hit_position = positions_by_key[self._document_key(hit)]
            for position in self._neighbor_positions(hit_position, len(source_docs), radius):
                selected_positions.add(position)
                if len(selected_positions) >= budget:
                    return self._order_expanded_neighbors(source_docs, selected_positions, hit_position)

        first_hit_position = positions_by_key[self._document_key(ranked_hits[0])]
        return self._order_expanded_neighbors(source_docs, selected_positions, first_hit_position)

    def _sibling_unit_paths(self, documents: list[Document], paths: set[str]) -> list[str]:
        parent_paths = {self._parent_path(path) for path in paths}
        siblings: list[str] = []
        seen: set[str] = set(paths)
        for document in documents:
            path = self._unit_key(document).path
            if path in seen:
                continue
            if self._parent_path(path) in parent_paths:
                siblings.append(path)
                seen.add(path)
        return siblings

    def _prefer_contiguous(self, selected: list[Document], source_docs: list[Document], budget: int) -> list[Document]:
        if len(selected) >= budget:
            return selected[:budget]
        selected_keys = {self._document_key(document) for document in selected}
        positions = [index for index, document in enumerate(source_docs) if self._document_key(document) in selected_keys]
        if not positions:
            return selected
        start = min(positions)
        end = min(len(source_docs), start + budget)
        return source_docs[start:end]

    def _evaluate_coverage(self, documents: list[Document], policy: RetrievalPolicy) -> CoverageReport:
        if not documents:
            return CoverageReport(
                sufficient=False,
                score=0.0,
                dominant_source=None,
                selected_sources=[],
                dominant_source_chunks=0,
                selected_chunks=0,
                total_available_chunks=len(self.documents),
                selected_units=0,
                total_available_units=sum(self._source_unit_counts.values()),
                continuity_ratio=0.0,
                structural_ratio=0.0,
                source_concentration=0.0,
                reason="no_documents",
            )

        source_counts: dict[str, int] = defaultdict(int)
        selected_units_by_source: dict[str, set[str]] = defaultdict(set)
        structural_chunks = 0
        for document in documents:
            source = self._source_key(document)
            source_counts[source] += 1
            selected_units_by_source[source].add(self._unit_key(document).path)
            if self._has_structural_metadata(document):
                structural_chunks += 1

        ranked_sources = sorted(source_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        dominant_source = ranked_sources[0][0]
        dominant_chunks = source_counts[dominant_source]
        selected_sources = [source for source, _count in ranked_sources[: max(1, policy.max_sources)]]
        selected_chunks = len(documents)
        selected_units = sum(len(paths) for paths in selected_units_by_source.values())
        evidence_units = {
            self._unit_key(document)
            for document in documents
            if (document.metadata.get("context_role") == "hit" or not document.metadata.get("context_role"))
        }
        total_units = sum(self._source_unit_counts.get(source, 0) for source in selected_sources)
        total_chunks = sum(len(self._documents_by_source.get(source, [])) for source in selected_sources)
        continuity_ratio = self._continuity_ratio(documents)
        structural_ratio = structural_chunks / max(selected_chunks, 1)
        source_concentration = dominant_chunks / max(selected_chunks, 1)
        unit_ratio = selected_units / max(total_units, 1)
        chunk_ratio = selected_chunks / max(total_chunks, 1)
        score = min(
            1.0,
            (0.35 * chunk_ratio)
            + (0.30 * unit_ratio)
            + (0.20 * continuity_ratio)
            + (0.15 * structural_ratio),
        )

        sufficient = self._is_coverage_sufficient(
            policy=policy,
            selected_chunks=selected_chunks,
            evidence_units=len(evidence_units),
            chunk_ratio=chunk_ratio,
            unit_ratio=unit_ratio,
            continuity_ratio=continuity_ratio,
            structural_ratio=structural_ratio,
        )
        reason = self._coverage_reason(
            sufficient=sufficient,
            policy=policy,
            selected_chunks=selected_chunks,
            evidence_units=len(evidence_units),
            chunk_ratio=chunk_ratio,
            unit_ratio=unit_ratio,
            continuity_ratio=continuity_ratio,
            structural_ratio=structural_ratio,
        )

        return CoverageReport(
            sufficient=sufficient,
            score=score,
            dominant_source=dominant_source,
            selected_sources=selected_sources,
            dominant_source_chunks=dominant_chunks,
            selected_chunks=selected_chunks,
            total_available_chunks=total_chunks,
            selected_units=selected_units,
            total_available_units=total_units,
            continuity_ratio=continuity_ratio,
            structural_ratio=structural_ratio,
            source_concentration=source_concentration,
            source_reports=self._source_reports(selected_sources, documents),
            reason=reason,
        )

    def _is_coverage_sufficient(
        self,
        *,
        policy: RetrievalPolicy,
        selected_chunks: int,
        evidence_units: int,
        chunk_ratio: float,
        unit_ratio: float,
        continuity_ratio: float,
        structural_ratio: float,
    ) -> bool:
        if selected_chunks < policy.coverage_min_chunks:
            return False
        if policy.intent in {QueryIntent.ENUMERATION, QueryIntent.AGGREGATION}:
            return chunk_ratio >= policy.coverage_min_ratio or unit_ratio >= policy.coverage_min_ratio
        if self._is_broad_intent(policy.intent):
            return chunk_ratio >= policy.coverage_min_ratio or selected_chunks >= policy.context_limit
        if policy.intent == QueryIntent.COMPARISON:
            return (
                unit_ratio >= min(policy.coverage_min_ratio, 0.4)
                or evidence_units >= policy.coverage_min_chunks
            )
        return chunk_ratio >= policy.coverage_min_ratio or continuity_ratio > 0.0 or structural_ratio > 0.0

    def _coverage_reason(
        self,
        *,
        sufficient: bool,
        policy: RetrievalPolicy,
        selected_chunks: int,
        evidence_units: int,
        chunk_ratio: float,
        unit_ratio: float,
        continuity_ratio: float,
        structural_ratio: float,
    ) -> str:
        if sufficient:
            return "coverage_ok"
        if selected_chunks < policy.coverage_min_chunks:
            return "not_enough_chunks"
        if policy.intent in {QueryIntent.ENUMERATION, QueryIntent.AGGREGATION} and unit_ratio < policy.coverage_min_ratio:
            return "structural_unit_coverage_low"
        if policy.intent == QueryIntent.COMPARISON and evidence_units < policy.coverage_min_chunks:
            return "comparison_evidence_units_low"
        if self._is_broad_intent(policy.intent) and chunk_ratio < policy.coverage_min_ratio:
            return "document_coverage_low"
        if structural_ratio <= 0.0:
            return "missing_structural_metadata"
        if continuity_ratio <= 0.0:
            return "continuity_low"
        return "coverage_ratio_low"

    def _source_reports(self, selected_sources: list[str], selected_documents: list[Document]) -> list[SourceCoverage]:
        reports: list[SourceCoverage] = []
        for source in selected_sources:
            source_selected = [document for document in selected_documents if self._source_key(document) == source]
            total_docs = self._documents_by_source.get(source, [])
            selected_units = {self._unit_key(document).path for document in source_selected}
            reports.append(
                SourceCoverage(
                    source=source,
                    selected_chunks=len(source_selected),
                    total_chunks=len(total_docs),
                    selected_units=len(selected_units),
                    total_units=self._source_unit_counts.get(source, 0),
                    continuity_ratio=self._continuity_ratio(source_selected),
                    structural_ratio=sum(1 for document in source_selected if self._has_structural_metadata(document))
                    / max(len(source_selected), 1),
                )
            )
        return reports

    def _group_by_source(self, documents: list[Document]) -> dict[str, list[Document]]:
        grouped: dict[str, list[tuple[int, Document]]] = defaultdict(list)
        for index, document in enumerate(documents):
            grouped[self._source_key(document)].append((index, document))
        return {
            source: [document for _index, document in sorted(items, key=self._sort_key)]
            for source, items in grouped.items()
        }

    def _rank_sources(self, query: str, documents: list[Document]) -> list[str]:
        query_tokens = self._tokens(query)
        stats: dict[str, dict[str, float]] = {}
        for document in documents:
            source = self._source_key(document)
            bucket = stats.setdefault(source, {"count": 0.0, "score_sum": 0.0, "best_score": 0.0, "match": 0.0})
            score = self._score_value(document)
            bucket["count"] += 1
            bucket["score_sum"] += score
            bucket["best_score"] = max(bucket["best_score"], score)
            bucket["match"] += len(query_tokens & self._document_descriptor_tokens(document))

        ranked = sorted(
            stats.items(),
            key=lambda item: (
                item[1]["best_score"],
                item[1]["match"],
                item[1]["score_sum"],
                item[1]["count"],
                item[0],
            ),
            reverse=True,
        )
        return [source for source, _stats in ranked]

    def _interleave_sources(self, expanded_by_source: dict[str, list[Document]], selected_sources: list[str]) -> list[Document]:
        expanded: list[Document] = []
        max_rounds = max((len(documents) for documents in expanded_by_source.values()), default=0)
        for index in range(max_rounds):
            for source in selected_sources:
                source_docs = expanded_by_source.get(source, [])
                if index < len(source_docs):
                    expanded.append(source_docs[index])
        return expanded

    def _dedupe_and_sort(
        self,
        documents: Iterable[Document],
    ) -> list[Document]:
        return self._dedupe_preserving_order(sorted(documents, key=self._sort_document))

    def _dedupe_by_retrieval_priority(self, documents: Iterable[Document]) -> list[Document]:
        return self._dedupe_preserving_order(sorted(documents, key=self._retrieval_priority_key))

    def _dedupe_preserving_order(self, documents: Iterable[Document]) -> list[Document]:
        seen: set[str] = set()
        unique_documents: list[Document] = []
        for document in documents:
            key = self._document_key(document)
            if key in seen:
                continue
            seen.add(key)
            unique_documents.append(clone_document(document))
        return unique_documents

    def _retrieved_by_key(self, candidates: Iterable[Document]) -> dict[str, Document]:
        return {self._document_key(document): document for document in candidates}

    def _mark_hits(self, documents: Iterable[Document]) -> list[Document]:
        marked: list[Document] = []
        for document in documents:
            cloned = clone_document(document)
            cloned.metadata["context_role"] = "hit"
            marked.append(cloned)
        return marked

    def _context_documents(
        self,
        expanded_documents: Iterable[Document],
        retrieved_by_key: dict[str, Document],
        source_candidates: list[Document],
    ) -> list[Document]:
        source_hits = sorted(source_candidates, key=self._retrieval_priority_key)
        primary_hit = source_hits[0] if source_hits else None
        primary_rank = primary_hit.metadata.get("retrieval_rank") if primary_hit is not None else None
        primary_chunk_id = self._document_key(primary_hit) if primary_hit is not None else None
        primary_chunk_index = self._chunk_index(primary_hit) if primary_hit is not None else None

        context_documents: list[Document] = []
        for document in expanded_documents:
            key = self._document_key(document)
            if key in retrieved_by_key:
                cloned = clone_document(retrieved_by_key[key])
                cloned.metadata["context_role"] = "hit"
            else:
                cloned = clone_document(document)
                cloned.metadata["context_role"] = "structural" if self._has_structural_metadata(cloned) else "neighbor"
                if primary_chunk_id is not None:
                    cloned.metadata["expanded_from_chunk_id"] = primary_chunk_id
                if primary_rank is not None:
                    cloned.metadata["expanded_from_retrieval_rank"] = primary_rank
                neighbor_chunk_index = self._chunk_index(cloned)
                if primary_chunk_index is not None and neighbor_chunk_index is not None:
                    cloned.metadata["neighbor_distance"] = abs(neighbor_chunk_index - primary_chunk_index)
            context_documents.append(cloned)
        return context_documents

    def _source_budget(self, policy: RetrievalPolicy, source_count: int) -> int:
        shared_budget = max(1, policy.context_limit // max(source_count, 1))
        return max(1, min(policy.source_chunk_limit, shared_budget))

    def _trim_positions_around_hits(self, positions: list[int], hits: list[int], budget: int) -> list[int]:
        if not hits:
            return positions[:budget]
        center = min(hits)
        return sorted(positions, key=lambda position: (abs(position - center), position))[:budget]

    def _continuity_ratio(self, documents: list[Document]) -> float:
        if len(documents) <= 1:
            return 1.0 if documents else 0.0
        ratios: list[float] = []
        for _source, source_docs in self._documents_grouped_for_metric(documents).items():
            indexes = sorted(self._chunk_index(document) for document in source_docs if self._chunk_index(document) is not None)
            if len(indexes) <= 1:
                ratios.append(1.0)
                continue
            consecutive = sum(1 for left, right in zip(indexes, indexes[1:]) if right == left + 1)
            ratios.append(consecutive / (len(indexes) - 1))
        return sum(ratios) / max(len(ratios), 1)

    def _documents_grouped_for_metric(self, documents: list[Document]) -> dict[str, list[Document]]:
        grouped: dict[str, list[Document]] = defaultdict(list)
        for document in documents:
            grouped[self._source_key(document)].append(document)
        return grouped

    def _has_structural_metadata(self, document: Document) -> bool:
        metadata = document.metadata
        if metadata.get("structural_type") or metadata.get("hierarchy_path") or metadata.get("header_path"):
            return True
        if metadata.get("headings") or self._looks_like_list_or_table(document):
            return True
        return any(key in metadata for key in ("page_number", "slide_number", "sheet_name", "json_path"))

    def _unit_key(self, document: Document) -> _StructureKey:
        source = self._source_key(document)
        metadata = document.metadata
        path = self._first_non_empty(
            metadata.get("hierarchy_path"),
            metadata.get("header_path"),
            self._headings_path(metadata.get("headings")),
            metadata.get("json_path"),
            self._page_or_slide_path(metadata),
            self._table_path(metadata),
            metadata.get("document_id"),
            source,
        )
        return _StructureKey(source=source, path=str(path))

    def _page_or_slide_path(self, metadata: dict[str, Any]) -> str:
        if metadata.get("page_number") not in {None, ""}:
            return f"page:{metadata['page_number']}"
        if metadata.get("slide_number") not in {None, ""}:
            return f"slide:{metadata['slide_number']}"
        return ""

    def _table_path(self, metadata: dict[str, Any]) -> str:
        if metadata.get("sheet_name") not in {None, ""}:
            return f"sheet:{metadata['sheet_name']}"
        return "table" if metadata.get("structural_type") == "table" else ""

    def _headings_path(self, headings: Any) -> str:
        if isinstance(headings, (list, tuple)):
            return " > ".join(str(part).strip() for part in headings if str(part).strip())
        return str(headings or "")

    def _parent_path(self, path: str) -> str:
        if " > " in path:
            return path.rsplit(" > ", 1)[0]
        if "." in path and path.startswith("$"):
            return path.rsplit(".", 1)[0]
        if "[" in path and path.startswith("$"):
            return path.rsplit("[", 1)[0]
        if ":" in path:
            return path.split(":", 1)[0]
        return ""

    def _document_descriptor_tokens(self, document: Document) -> set[str]:
        metadata = document.metadata
        descriptor = " ".join(
            str(value)
            for value in (
                metadata.get("source"),
                metadata.get("source_name"),
                metadata.get("file_type"),
                metadata.get("hierarchy_path"),
                metadata.get("header_path"),
                metadata.get("sheet_name"),
                metadata.get("json_path"),
            )
            if value
        )
        return self._tokens(descriptor)

    def _mentions_source_scope(self, query: str, source: str) -> bool:
        query_tokens = self._tokens(query)
        return bool(query_tokens & self._tokens(source))

    def _is_broad_intent(self, intent: QueryIntent) -> bool:
        return intent in {QueryIntent.SUMMARY, QueryIntent.BROAD_SEMANTIC}

    def _looks_like_list_or_table(self, document: Document) -> bool:
        structural_type = str(document.metadata.get("structural_type", "")).lower()
        return structural_type == "table" or bool(
            BULLET_OR_NUMBERED_LIST_PATTERN.search(document.page_content)
            or MARKDOWN_TABLE_ROW_PATTERN.search(document.page_content)
        )

    def _source_key(self, document: Document) -> str:
        source = str(document.metadata.get("source", "")).strip()
        return source or "unknown"

    def _document_key(self, document: Document) -> str:
        if document is None:
            return ""
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        document_id = document.metadata.get("document_id")
        chunk_index = document.metadata.get("chunk_index")
        if document_id not in {None, ""} and chunk_index not in {None, ""}:
            return f"document:{document_id}|chunk:{chunk_index}"
        source = self._source_key(document)
        if source != "unknown" and chunk_index not in {None, ""}:
            return f"source:{source}|chunk:{chunk_index}"
        return "|".join(
            [
                source,
                str(chunk_index or ""),
                str(document.metadata.get("hierarchy_path", "")),
                str(document.metadata.get("header_path", "")),
                document.page_content.strip(),
            ]
        )

    def _score_value(self, document: Document) -> float:
        raw_score = document.metadata.get("retrieval_score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0
        return score if math.isfinite(score) else 0.0

    def _retrieval_priority_key(self, document: Document) -> tuple[float, int, str, int, str]:
        retrieval_rank = self._int_metadata(document, "retrieval_rank", default=10**9)
        chunk_index = self._chunk_index(document)
        chunk_value = chunk_index if chunk_index is not None else 10**9
        return (
            -self._score_value(document),
            retrieval_rank,
            self._source_key(document),
            chunk_value,
            self._document_key(document),
        )

    def _order_expanded_neighbors(
        self,
        source_docs: list[Document],
        selected_positions: set[int],
        primary_position: int,
    ) -> list[Document]:
        if not selected_positions:
            return []
        ordered_positions = [primary_position] if primary_position in selected_positions else []
        ordered_positions.extend(position for position in sorted(selected_positions) if position != primary_position)
        return [source_docs[index] for index in ordered_positions]

    def _neighbor_positions(self, center: int, document_count: int, radius: int) -> list[int]:
        positions = [center]
        for distance in range(1, radius + 1):
            left = center - distance
            right = center + distance
            if left >= 0:
                positions.append(left)
            if right < document_count:
                positions.append(right)
        return positions

    def _sort_key(self, item: tuple[int, Document]) -> tuple[int, str]:
        index, document = item
        chunk_index = self._chunk_index(document)
        chunk_value = chunk_index if chunk_index is not None else index
        return (chunk_value, self._document_key(document))

    def _sort_document(self, document: Document) -> tuple[int, str]:
        chunk_index = self._chunk_index(document)
        chunk_value = chunk_index if chunk_index is not None else 10**9
        return (chunk_value, self._document_key(document))

    def _chunk_index(self, document: Document) -> int | None:
        try:
            return int(document.metadata.get("chunk_index"))
        except (TypeError, ValueError):
            return None

    def _int_metadata(self, document: Document, key: str, *, default: int) -> int:
        try:
            return int(document.metadata.get(key))
        except (TypeError, ValueError):
            return default

    def _tokens(self, text: str) -> set[str]:
        return set(TOKEN_PATTERN.findall(text.casefold()))

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return ""
