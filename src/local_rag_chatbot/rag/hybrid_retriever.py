from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
import importlib.util
import logging
import math
import re
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from local_rag_chatbot.observability import metrics


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def document_key(document: Document) -> str:
    source = str(document.metadata.get("source", "")).strip()
    chunk_index = document.metadata.get("chunk_index")
    header_path = str(document.metadata.get("header_path", "")).strip()
    content_hash = sha1(document.page_content.encode("utf-8")).hexdigest()
    if source or header_path or chunk_index is not None:
        return f"{source}:{chunk_index}:{header_path}:{content_hash}"
    return content_hash


def normalize_score(score: float, minimum: float, maximum: float) -> float:
    if not all(math.isfinite(value) for value in (score, minimum, maximum)):
        return 0.0
    if maximum <= minimum:
        return 1.0
    normalized = (score - minimum) / (maximum - minimum)
    return max(0.0, min(1.0, normalized))


def clone_document(document: Document) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
    )


@dataclass(frozen=True)
class RetrievalTuning:
    dense_k: int = 10
    dense_fetch_k: int = 20
    sparse_k: int = 10
    candidate_k: int = 10
    final_k: int = 5
    dense_lambda_mult: float = 0.5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    metadata_weight: float = 0.10
    rrf_k: int = 60


@dataclass(frozen=True)
class RetrievalEvidence:
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_rrf_score: float = 0.0
    reranker_score: float | None = None
    metadata_boost: float = 0.0
    final_ranking_score: float = 0.0

    def as_metadata(self) -> dict[str, Any]:
        return {
            "retrieval_dense_score": self.dense_score,
            "retrieval_dense_rank": self.dense_rank,
            "retrieval_sparse_score": self.sparse_score,
            "retrieval_sparse_rank": self.sparse_rank,
            "retrieval_fused_rrf_score": round(self.fused_rrf_score, 6),
            "retrieval_reranker_score": self.reranker_score,
            "retrieval_metadata_boost": round(self.metadata_boost, 6),
            "retrieval_final_ranking_score": round(self.final_ranking_score, 6),
        }


@dataclass(frozen=True)
class MetadataBoostRule:
    trigger_terms: frozenset[str]
    match_terms: frozenset[str]
    boost: float


@dataclass(frozen=True)
class RerankingFeatureState:
    configured_mode: str
    active: bool
    status: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured_mode": self.configured_mode,
            "active": self.active,
            "status": self.status,
            "reason": self.reason,
        }


class MetadataBoostScorer:
    def __init__(self) -> None:
        self.rules: tuple[MetadataBoostRule, ...] = ()

    def score(self, query: str, document: Document) -> float:
        header_text = self._header_text(document)
        if not header_text:
            return 0.0

        header_tokens = set(tokenize(header_text))
        query_tokens = set(tokenize(query))
        query_text = query.lower()

        score = min(0.3, 0.06 * len(query_tokens & header_tokens))
        header_text_lower = header_text.lower()

        for rule in self.rules:
            if not query_tokens & rule.trigger_terms and not any(term in query_text for term in rule.trigger_terms):
                continue

            if any(term in header_text_lower for term in rule.match_terms):
                score += rule.boost

        if query_text.strip() and query_text.strip() in header_text_lower:
            score += 0.4

        return min(score, 1.0)

    def _header_text(self, document: Document) -> str:
        values = [
            document.metadata.get("header_path", ""),
            document.metadata.get("header_1", ""),
            document.metadata.get("header_2", ""),
            document.metadata.get("header_3", ""),
            document.metadata.get("Header 1", ""),
            document.metadata.get("Header 2", ""),
            document.metadata.get("Header 3", ""),
        ]
        return " ".join(str(value).strip() for value in values if str(value).strip())


class SimpleBM25Index:
    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized_docs = [self._document_tokens(document) for document in documents]
        self._doc_lengths = [len(tokens) for tokens in self._tokenized_docs]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        self._doc_freqs: Counter[str] = Counter()
        for tokens in self._tokenized_docs:
            self._doc_freqs.update(set(tokens))
        self._doc_count = len(documents)

    def search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        if not self.documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[Document, float]] = []
        for document, tokens, doc_length in zip(self.documents, self._tokenized_docs, self._doc_lengths):
            token_counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = token_counts.get(term)
                if not frequency:
                    continue
                score += self._score_term(term, frequency, doc_length)
            if score > 0:
                scores.append((document, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]

    def _document_tokens(self, document: Document) -> list[str]:
        header_bits = [
            str(document.metadata.get("header_path", "")),
            str(document.metadata.get("header_1", "")),
            str(document.metadata.get("header_2", "")),
            str(document.metadata.get("header_3", "")),
        ]
        return tokenize(" ".join([document.page_content, *header_bits]))

    def _score_term(self, term: str, frequency: int, doc_length: int) -> float:
        if self._doc_count == 0 or self._avg_doc_length == 0:
            return 0.0

        doc_frequency = self._doc_freqs.get(term, 0)
        idf = math.log(1 + ((self._doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5)))
        denominator = frequency + self.k1 * (1 - self.b + self.b * (doc_length / self._avg_doc_length))
        return idf * (frequency * (self.k1 + 1)) / denominator


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, allow_lexical_fallback: bool = False) -> None:
        self.model_name = model_name
        self.allow_lexical_fallback = allow_lexical_fallback
        self._model: Any | None = None
        self._load_error: Exception | None = None
        self._warned = False

    def rerank(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            if not self.allow_lexical_fallback:
                return []
            return self._fallback_scores(query, documents)

        pairs = [(query, self._document_text(document)) for document in documents]
        predictions = model.predict(pairs)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        scores = [float(score) for score in predictions]
        return list(zip(documents, scores))

    def load(self) -> bool:
        return self._load_model() is not None

    def _load_model(self) -> Any | None:
        if self._model is not None or self._load_error is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception as exc:  # pragma: no cover - import/model download fallback
            self._load_error = exc
            self._model = None

        return self._model

    @property
    def available(self) -> bool:
        return self._model is not None or (
            self._load_error is None and importlib.util.find_spec("sentence_transformers") is not None
        )

    def _fallback_scores(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        query_tokens = set(tokenize(query))
        scored_documents: list[tuple[Document, float]] = []

        for document in documents:
            doc_text = self._document_text(document)
            doc_tokens = set(tokenize(doc_text))
            overlap = len(query_tokens & doc_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            header_overlap = len(query_tokens & set(tokenize(self._header_text(document))))
            lexical_boost = min(0.1, 0.02 * header_overlap)
            scored_documents.append((document, coverage + lexical_boost))

        return scored_documents

    def _document_text(self, document: Document) -> str:
        header_bits = [
            str(document.metadata.get("header_path", "")),
            str(document.metadata.get("header_1", "")),
            str(document.metadata.get("header_2", "")),
            str(document.metadata.get("header_3", "")),
        ]
        return " ".join([document.page_content, *header_bits]).strip()

    def _header_text(self, document: Document) -> str:
        return " ".join(
            str(value).strip()
            for value in (
                document.metadata.get("header_path", ""),
                document.metadata.get("header_1", ""),
                document.metadata.get("header_2", ""),
                document.metadata.get("header_3", ""),
                document.metadata.get("Header 1", ""),
                document.metadata.get("Header 2", ""),
                document.metadata.get("Header 3", ""),
            )
            if str(value).strip()
        )


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    documents: list[Document]
    tuning: RetrievalTuning = RetrievalTuning()
    reranking_mode: str = "auto"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_allow_lexical_fallback: bool = False

    _bm25_index: SimpleBM25Index = PrivateAttr()
    _reranker: CrossEncoderReranker | None = PrivateAttr(default=None)
    _metadata_boost: MetadataBoostScorer = PrivateAttr()
    _reranking_state: RerankingFeatureState = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._bm25_index = SimpleBM25Index(self.documents)
        self._metadata_boost = MetadataBoostScorer()
        self._configure_reranker()

    @property
    def reranking_active(self) -> bool:
        return self._reranking_state.active

    @property
    def reranking_status(self) -> dict[str, Any]:
        return self._reranking_state.as_dict()

    def _configure_reranker(self) -> None:
        mode = self.reranking_mode
        self._reranking_state = RerankingFeatureState(
            configured_mode=mode,
            active=False,
            status="initializing",
            reason=None,
        )

        if mode == "false":
            logger.info("Reranking is disabled by configuration. Standard hybrid retrieval will be used.")
            self._reranking_state = RerankingFeatureState(
                configured_mode=mode,
                active=False,
                status="disabled",
                reason="Reranking was disabled by configuration.",
            )
            return

        dependency_available = importlib.util.find_spec("sentence_transformers") is not None
        if not dependency_available:
            reason = "The optional sentence-transformers dependency is not installed."
            if mode == "true":
                logger.warning(
                    'RERANKING_MODE=true, but the optional sentence-transformers dependency is not installed. '
                    'Continuing without reranking. Install it with: pip install -e ".[rerank]"'
                )
            else:
                logger.info(
                    "Reranking is unavailable because the optional sentence-transformers dependency is not "
                    'installed. Continuing with standard hybrid retrieval. Install the rerank extra to enable '
                    'it: pip install -e ".[rerank]"'
                )
            self._reranking_state = RerankingFeatureState(
                configured_mode=mode,
                active=False,
                status="dependency_missing",
                reason=reason,
            )
            return

        reranker = CrossEncoderReranker(
            self.reranker_model,
            allow_lexical_fallback=self.reranker_allow_lexical_fallback,
        )
        if reranker.load():
            self._reranker = reranker
            logger.info("Reranker loaded successfully. Document reranking is active.")
            self._reranking_state = RerankingFeatureState(
                configured_mode=mode,
                active=True,
                status="active",
                reason=None,
            )
            return

        logger.warning("The reranker could not be loaded. Continuing with standard hybrid retrieval.")
        if reranker._load_error is not None:
            logger.debug(
                "Reranker load failure details",
                exc_info=(
                    type(reranker._load_error),
                    reranker._load_error,
                    reranker._load_error.__traceback__,
                ),
            )
        self._reranking_state = RerankingFeatureState(
            configured_mode=mode,
            active=False,
            status="load_failed",
            reason="The reranker model could not be loaded.",
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any | None = None,
    ) -> list[Document]:
        return self.retrieve(query)

    def retrieve(
        self,
        query: str,
        *,
        candidate_limit: int | None = None,
        final_limit: int | None = None,
        dense_k: int | None = None,
        dense_fetch_k: int | None = None,
        sparse_k: int | None = None,
    ) -> list[Document]:
        with metrics.timer("retrieval_dense_seconds"):
            dense_docs = self._dense_candidates(query, k=dense_k, fetch_k=dense_fetch_k)
        with metrics.timer("retrieval_sparse_seconds"):
            sparse_results = self._bm25_index.search(query, sparse_k or self.tuning.sparse_k)

        if not dense_docs and not sparse_results:
            return []

        with metrics.timer("retrieval_fusion_seconds"):
            fused_documents = self._fuse_candidates(
                query,
                dense_docs,
                sparse_results,
                candidate_limit=candidate_limit,
            )
        with metrics.timer("retrieval_reranking_seconds"):
            reranked = self._rerank(query, fused_documents)
        limit = final_limit if final_limit is not None else self.tuning.final_k
        return self._with_retrieval_ranks(reranked[:limit])

    def _dense_candidates(
        self,
        query: str,
        *,
        k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[Document]:
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k or self.tuning.dense_k,
                "fetch_k": fetch_k or self.tuning.dense_fetch_k,
                "lambda_mult": self.tuning.dense_lambda_mult,
            },
        )
        return list(retriever.invoke(query))

    def _fuse_candidates(
        self,
        query: str,
        dense_docs: list[Document],
        sparse_results: list[tuple[Document, float]],
        *,
        candidate_limit: int | None = None,
    ) -> list[Document]:
        ranked_documents: dict[str, dict[str, Any]] = {}

        def register(document: Document, source: str, rank: int, *, raw_score: float | None = None) -> None:
            key = document_key(document)
            bucket = ranked_documents.setdefault(
                key,
                {
                    "document": clone_document(document),
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "source_hits": set(),
                },
            )
            bucket["source_hits"].add(source)
            bucket[f"{source}_rank"] = rank
            if source == "sparse":
                bucket["sparse_score"] = raw_score
            elif source == "dense":
                bucket["dense_score"] = raw_score
            bucket.setdefault("source_documents", {})[source] = document

        for rank, document in enumerate(dense_docs, start=1):
            register(document, "dense", rank, raw_score=self._metadata_float(document, "retrieval_dense_score"))

        for rank, sparse_item in enumerate(sparse_results, start=1):
            if isinstance(sparse_item, tuple):
                document, sparse_score = sparse_item
            else:
                document = sparse_item
                sparse_score = self._metadata_float(document, "retrieval_sparse_score")
            register(document, "sparse", rank, raw_score=sparse_score)

        candidates: list[tuple[Document, float, float, RetrievalEvidence]] = []
        for bucket in ranked_documents.values():
            document = bucket["document"]
            dense_rank = bucket["dense_rank"]
            dense_raw_score = bucket["dense_score"]
            sparse_rank = bucket["sparse_rank"]
            sparse_raw_score = bucket["sparse_score"]

            dense_score = self._rrf_score(dense_rank)
            sparse_score = self._rrf_score(sparse_rank)
            fused_rank_score = (
                self.tuning.dense_weight * dense_score
                + self.tuning.sparse_weight * sparse_score
            )
            metadata_score = self._metadata_boost.score(query, document)
            evidence = RetrievalEvidence(
                dense_score=dense_raw_score,
                dense_rank=dense_rank,
                sparse_score=sparse_raw_score,
                sparse_rank=sparse_rank,
                fused_rrf_score=fused_rank_score,
                metadata_boost=metadata_score,
            )
            candidates.append((document, fused_rank_score, metadata_score, evidence))

        if not candidates:
            return []

        rank_scores = [score for _document, score, _metadata, _evidence in candidates]
        min_rank_score = min(rank_scores)
        max_rank_score = max(rank_scores)

        scored_candidates: list[tuple[Document, float]] = []
        for document, rank_score, metadata_score, evidence in candidates:
            normalized_rank_score = normalize_score(rank_score, min_rank_score, max_rank_score)
            final_score = (
                (1.0 - self.tuning.metadata_weight) * normalized_rank_score
                + self.tuning.metadata_weight * metadata_score
            )
            cloned_document = clone_document(document)
            cloned_document.metadata.update(
                RetrievalEvidence(
                    dense_score=evidence.dense_score,
                    dense_rank=evidence.dense_rank,
                    sparse_score=evidence.sparse_score,
                    sparse_rank=evidence.sparse_rank,
                    fused_rrf_score=evidence.fused_rrf_score,
                    metadata_boost=metadata_score,
                    final_ranking_score=final_score,
                ).as_metadata()
            )
            cloned_document.metadata["retrieval_score"] = round(final_score, 6)
            cloned_document.metadata["retrieval_rank_score"] = round(normalized_rank_score, 6)
            scored_candidates.append((cloned_document, final_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        limit = candidate_limit if candidate_limit is not None else self.tuning.candidate_k
        return [document for document, _score in scored_candidates[:limit]]

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if self._reranker is None:
            return documents

        reranked_scores = self._reranker.rerank(query, documents)
        if not reranked_scores:
            return documents

        base_scores = [score for _document, score in reranked_scores]
        minimum = min(base_scores)
        maximum = max(base_scores)

        scored_documents: list[tuple[Document, float]] = []
        for document, base_score in reranked_scores:
            metadata_score = self._metadata_boost.score(query, document)
            normalized_rerank = normalize_score(base_score, minimum, maximum)
            final_score = (
                (1.0 - self.tuning.metadata_weight) * normalized_rerank
                + self.tuning.metadata_weight * metadata_score
            )
            cloned_document = clone_document(document)
            cloned_document.metadata["retrieval_reranker_score"] = round(float(base_score), 6)
            cloned_document.metadata["retrieval_final_ranking_score"] = round(final_score, 6)
            cloned_document.metadata["retrieval_score"] = round(final_score, 6)
            cloned_document.metadata["retrieval_rerank_score"] = round(float(base_score), 6)
            cloned_document.metadata["retrieval_metadata_boost"] = round(metadata_score, 6)
            scored_documents.append((cloned_document, final_score))

        scored_documents.sort(key=lambda item: item[1], reverse=True)
        return [document for document, _score in scored_documents]

    def _with_retrieval_ranks(self, documents: list[Document]) -> list[Document]:
        ranked_documents: list[Document] = []
        for rank, document in enumerate(documents, start=1):
            cloned_document = clone_document(document)
            cloned_document.metadata["retrieval_rank"] = rank
            ranked_documents.append(cloned_document)
        return ranked_documents

    def _rrf_score(self, rank: int | None) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return 1.0 / (self.tuning.rrf_k + rank)

    def _metadata_float(self, document: Document, key: str) -> float | None:
        value = document.metadata.get(key)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number
