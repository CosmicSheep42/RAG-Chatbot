"""Small, offline-capable benchmark runner for the fixed evaluation corpus.

Live model providers are intentionally injected by callers. Importing this module
never initializes a model, vector store, or network client.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Protocol

from langchain_core.documents import Document

from local_rag_chatbot.documents.document_splitter import DocumentSplitter
from local_rag_chatbot.documents.loaders import MarkdownLoader
from local_rag_chatbot.rag.hybrid_retriever import SimpleBM25Index
from local_rag_chatbot.rag.hybrid_retriever import HybridRetriever, RetrievalTuning
from local_rag_chatbot.rag.index_manager import RAG_VECTOR_NAMESPACE
from local_rag_chatbot.vectorstores.contracts import VectorStoreBackend

from .adapters.datapizza import load_examples as load_datapizza_examples
from .metrics import (
    abstention_result,
    citation_quality,
    citation_support,
    complete_passage_retrieval,
    complete_evidence_retrieval,
    fact_coverage,
    ndcg_at_k,
    passage_recall_at_k,
    passage_coverage_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

CONFIGURATIONS = ("standalone", "dense", "bm25", "hybrid", "hybrid-rerank", "full")
MODES = ("retrieval-only", "end-to-end")


def validate_configuration_mode(configuration: str, mode: str) -> None:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown benchmark configuration: {configuration}")
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    if configuration in {"standalone", "full"} and mode != "end-to-end":
        raise ValueError(f"{configuration} only supports end-to-end mode")


class Retriever(Protocol):
    def retrieve(self, question: str, limit: int) -> list[Document]: ...


class Answerer(Protocol):
    def answer(self, question: str, documents: list[Document]) -> tuple[str, list[str]]: ...


class ChatModelAnswerer:
    def __init__(self, chat_model: Any, model: str) -> None:
        self.chat_model = chat_model
        self.model = model

    def answer(self, question: str, documents: list[Document]) -> tuple[str, list[str]]:
        prompt = (
            "Answer the question using your general knowledge. If you are uncertain or do not know, "
            "say so rather than inventing details.\n\n"
            f"Question: {question}"
        )
        return self.chat_model.chat([{"role": "user", "content": prompt}], model=self.model).content.strip(), []


class EngineAnswerer:
    def __init__(self, engine: Any, event_loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.engine = engine
        self.event_loop = event_loop
        self.timing_name = "pipeline_seconds"
        self.last_retrieved: list[str] = []
        self.last_context: list[str] | None = None
        self.context_available = False
        self.last_route: str | None = None
        self.last_refused: bool | None = None

    def answer(self, question: str, documents: list[Document]) -> tuple[str, list[str]]:
        if self.event_loop is None:
            response = asyncio.run(self.engine.chat(question))
        else:
            response = self.event_loop.run_until_complete(self.engine.chat(question))
        self.last_retrieved = [source.source for source in response.sources]
        self.last_route = getattr(response, "route", None)
        self.last_refused = getattr(response, "refused", None)
        return response.result, [citation.source_name or "" for citation in response.citations]


class AnswerServiceAdapter:
    def __init__(self, answer_service: Any, policy_resolver: Any) -> None:
        self.answer_service = answer_service
        self.policy_resolver = policy_resolver

    def answer(self, question: str, documents: list[Document]) -> tuple[str, list[str]]:
        policy = self.policy_resolver.resolve(question)
        answer = self.answer_service.generate(question, documents, policy)
        return answer, []


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.index = SimpleBM25Index(documents)

    def retrieve(self, question: str, limit: int) -> list[Document]:
        return [document for document, _score in self.index.search(question, limit)]


class DenseRetriever:
    """Adapter over the application's configured vector-store contract."""

    def __init__(self, backend: VectorStoreBackend, namespace: str) -> None:
        self.backend = backend
        self.namespace = namespace

    def retrieve(self, question: str, limit: int) -> list[Document]:
        return [
            result.document
            for result in self.backend.search(self.namespace, question, k=limit, fetch_k=limit, strategy="similarity")
        ]


class ExistingRetrieverAdapter:
    """Give the production retriever the small interface used by the runner."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, question: str, limit: int) -> list[Document]:
        return self.retriever.retrieve(question, final_limit=limit)


def build_retriever(
    configuration: str,
    documents: list[Document],
    *,
    vector_backend: VectorStoreBackend | None = None,
    namespace: str = RAG_VECTOR_NAMESPACE,
    tuning: RetrievalTuning | None = None,
    reranker_model: str = "BAAI/bge-reranker-base",
    reranker_device: str = "cpu",
) -> Retriever | None:
    """Build an adapter without changing or duplicating production retrieval logic."""
    if configuration == "standalone":
        return None
    if configuration == "bm25":
        return BM25Retriever(documents)
    if vector_backend is None:
        raise ValueError(f"{configuration} requires an injected vector-store backend")
    if not vector_backend.index_exists(namespace):
        raise RuntimeError(f"The benchmark index namespace does not exist: {namespace}")
    if configuration == "dense":
        return DenseRetriever(vector_backend, namespace)
    if configuration == "full":
        raise ValueError(
            "full requires the production ChatbotEngine pipeline; use an injected engine adapter rather than "
            "the retrieval-only builder"
        )
    if configuration in {"hybrid", "hybrid-rerank"}:
        production_retriever = HybridRetriever(
            vector_backend=vector_backend,
            namespace=namespace,
            documents=documents,
            tuning=tuning or RetrievalTuning(),
            reranking_mode="true" if configuration == "hybrid-rerank" else "false",
            reranker_model=reranker_model,
            reranker_device=reranker_device,
        )
        if configuration == "hybrid-rerank" and not production_retriever.reranking_active:
            status = production_retriever.reranking_status
            raise ValueError(f"hybrid-rerank is unavailable: {status['reason'] or status['status']}")
        return ExistingRetrieverAdapter(production_retriever)
    raise ValueError(f"Unknown benchmark configuration: {configuration}")


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    question: str
    answerable: bool
    reference_answer: str | None
    relevant_documents: list[str]
    required_facts: list[str]
    passages: list[dict[str, Any]] | None = None
    difficulty: str | None = None


def load_corpus(corpus_dir: Path, *, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    documents: list[Document] = []
    loader = MarkdownLoader()
    splitter = DocumentSplitter(chunk_size, chunk_overlap)
    for path in sorted(corpus_dir.glob("*.md")):
        loaded = loader.load(
            path,
            base_metadata={
                "source": path.name,
                "document_id": path.name,
                "_source_text": path.read_text(encoding="utf-8"),
            },
        )
        documents.extend(Document(item.page_content, metadata=item.metadata) for item in loaded)
    return splitter.split(documents)


def load_examples(dataset_path: Path, split: str) -> list[BenchmarkExample]:
    if dataset_path.is_dir():
        return [
            BenchmarkExample(
                id=example.id,
                question=example.question,
                answerable=True,
                reference_answer=example.answer,
                relevant_documents=example.relevant_documents,
                required_facts=[],
                passages=[passage.__dict__ for passage in example.passages],
                difficulty=example.tier,
            )
            for example in load_datapizza_examples(dataset_path, split)
        ]
    examples = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["split"] == split:
                examples.append(
                    BenchmarkExample(
                        id=item["id"],
                        question=item["question"],
                        answerable=item["answerable"],
                        reference_answer=item["reference_answer"],
                        relevant_documents=item["relevant_documents"],
                        required_facts=item["required_facts"],
                        passages=item.get("evidence"),
                    )
                )
    return examples


def run_retrieval(
    examples: Iterable[BenchmarkExample],
    retriever: Retriever,
    *,
    configuration: str,
    split: str,
    top_k: int,
    metadata: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    examples = list(examples)
    metadata = dict(metadata or {})
    metadata.setdefault("experiment_mode", "retrieval-only")
    metadata.setdefault("question_count", len(examples))
    metadata.setdefault("passage_coverage_threshold", 0.8)
    records = _load_checkpoint_records(checkpoint_dir) if resume and checkpoint_dir else []
    if retry_failed:
        records = [record for record in records if record.get("status") != "failed"]
    completed_ids = {
        record.get("question_id") for record in records if record.get("status") == "success" or not retry_failed
    }
    retrieval_latencies = []
    for example in examples:
        if example.id in completed_ids:
            continue
        started = time.perf_counter()
        try:
            documents = retriever.retrieve(example.question, top_k)
        except Exception as exc:
            records.append(
                {
                    "question_id": example.id,
                    "question": example.question,
                    "difficulty": example.difficulty,
                    "gold_passages": example.passages,
                    "configuration": configuration,
                    "split": split,
                    "answerable": example.answerable,
                    "status": "failed",
                    "error": str(exc),
                    "timing": {"retrieval_seconds": time.perf_counter() - started},
                }
            )
            _persist_checkpoint(checkpoint_dir, records, metadata)
            continue
        latency = time.perf_counter() - started
        retrieval_latencies.append(latency)
        ranked = [
            {
                "document_id": str(document.metadata.get("source", "")),
                "rank": rank,
                "start_char": document.metadata.get("source_char_start"),
                "end_char": document.metadata.get("source_char_end"),
            }
            for rank, document in enumerate(documents, 1)
        ]
        coverage_threshold = float(metadata["passage_coverage_threshold"])
        records.append(
            {
                "question_id": example.id,
                "question": example.question,
                "difficulty": example.difficulty,
                "gold_passages": example.passages,
                "configuration": configuration,
                "split": split,
                "answerable": example.answerable,
                "retrieved_documents": ranked,
                "metrics": {
                    "recall_at_k": _json_metric(recall_at_k(ranked, example.relevant_documents, top_k)),
                    "mrr_at_k": _json_metric(reciprocal_rank_at_k(ranked, example.relevant_documents, top_k)),
                    "ndcg_at_k": _json_metric(ndcg_at_k(ranked, example.relevant_documents, top_k)),
                    "passage_recall_at_k": (
                        _json_metric(passage_recall_at_k(ranked, example.passages or [], top_k))
                        if example.passages
                        else None
                    ),
                    "passage_coverage_at_k": (
                        _json_metric(
                            passage_coverage_at_k(ranked, example.passages or [], top_k, threshold=coverage_threshold)
                        )
                        if example.passages
                        else None
                    ),
                    "complete_passage_evidence": (
                        complete_passage_retrieval(ranked, example.passages or [], top_k, threshold=coverage_threshold)
                        if example.passages
                        else None
                    ),
                    "complete_evidence": complete_evidence_retrieval(ranked, example.relevant_documents, top_k),
                },
                "timing": {"retrieval_seconds": latency},
                "status": "success",
            }
        )
        _persist_checkpoint(checkpoint_dir, records, metadata)
    if not retrieval_latencies:
        retrieval_latencies = [
            float(record.get("timing", {}).get("retrieval_seconds", 0))
            for record in records
            if record.get("status") == "success" and "retrieval_seconds" in record.get("timing", {})
        ]
    summary = _summary(records, retrieval_latencies)
    summary["passage_coverage_threshold"] = metadata["passage_coverage_threshold"]
    return {
        "metadata": metadata or {},
        "configuration": configuration,
        "split": split,
        "records": records,
        "summary": summary,
    }


def run_end_to_end(
    examples: Iterable[BenchmarkExample],
    retriever: Retriever | None,
    answerer: Answerer,
    *,
    configuration: str,
    split: str,
    top_k: int,
    metadata: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    examples = list(examples)
    metadata = dict(metadata or {})
    metadata.setdefault("experiment_mode", "end-to-end")
    metadata.setdefault("question_count", len(examples))
    metadata.setdefault("passage_coverage_threshold", 0.8)
    records = _load_checkpoint_records(checkpoint_dir) if resume and checkpoint_dir else []
    if retry_failed:
        records = [record for record in records if record.get("status") != "failed"]
    completed_ids = {
        record.get("question_id") for record in records if record.get("status") == "success" or not retry_failed
    }
    retrieval_latencies = []
    for example in examples:
        if example.id in completed_ids:
            continue
        started = time.perf_counter()
        try:
            documents = retriever.retrieve(example.question, top_k) if retriever else []
            retrieval_seconds = time.perf_counter() - started
            answer_started = time.perf_counter()
            answer, citations = answerer.answer(example.question, documents)
            answer_seconds = time.perf_counter() - answer_started
        except Exception as exc:
            records.append(
                {
                    "question_id": example.id,
                    "question": example.question,
                    "difficulty": example.difficulty,
                    "gold_passages": example.passages,
                    "configuration": configuration,
                    "split": split,
                    "status": "failed",
                    "error": str(exc),
                    "timing": {"retrieval_seconds": time.perf_counter() - started},
                }
            )
            _persist_checkpoint(checkpoint_dir, records, metadata)
            continue
        retrieved = getattr(
            answerer,
            "last_retrieved",
            [str(document.metadata.get("source", "")) for document in documents],
        )
        context_available = getattr(answerer, "context_available", None)
        context = getattr(answerer, "last_context", None)
        if context_available is None:
            context_available = True
        if context_available and context is None:
            context = [document.page_content for document in documents]
        if not getattr(answerer, "timing_name", None):
            retrieval_latencies.append(retrieval_seconds)
        timing = {"retrieval_seconds": retrieval_seconds, "generation_seconds": answer_seconds}
        timing_name = getattr(answerer, "timing_name", None)
        if timing_name:
            timing = {timing_name: answer_seconds}
        citation_metrics = citation_quality(citations, retrieved, example.relevant_documents) if citations else None
        records.append(
            {
                "question_id": example.id,
                "question": example.question,
                "difficulty": example.difficulty,
                "gold_passages": example.passages,
                "configuration": configuration,
                "split": split,
                "answerable": example.answerable,
                "retrieved_documents": retrieved,
                "selected_context": context if context_available else None,
                "context_available": context_available,
                "answer": answer,
                "citations": citations,
                "route": getattr(answerer, "last_route", None),
                "refused": getattr(answerer, "last_refused", None),
                "metrics": {
                    "fact_coverage": (
                        _json_metric(fact_coverage(answer, example.required_facts)) if example.answerable else None
                    ),
                    "evidence_availability": (
                        _json_metric(fact_coverage(" ".join(context or []), example.required_facts))
                        if context_available and example.required_facts
                        else None
                    ),
                    "citation_validity": citation_metrics["validity"] if citation_metrics else None,
                    "citation_relevance": citation_metrics["relevance"] if citation_metrics else None,
                    "citation_support": citation_support(citations, example.relevant_documents) if citations else None,
                    "groundedness": "manual_review_required",
                    "abstention": abstention_result(answer, example.answerable),
                },
                "manual_review": {
                    "answer": answer,
                    "reference_answer": example.reference_answer,
                    "required_facts": example.required_facts,
                    "retrieved_context": context if context_available else None,
                },
                "timing": timing,
                "status": "success",
            }
        )
        _persist_checkpoint(checkpoint_dir, records, metadata)
    if not retrieval_latencies:
        retrieval_latencies = [
            float(record.get("timing", {}).get("retrieval_seconds", 0))
            for record in records
            if record.get("status") == "success" and "retrieval_seconds" in record.get("timing", {})
        ]
    summary = _summary(records, retrieval_latencies)
    summary["passage_coverage_threshold"] = metadata["passage_coverage_threshold"]
    return {
        "metadata": metadata or {},
        "configuration": configuration,
        "split": split,
        "records": records,
        "summary": summary,
    }


def _summary(records: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    metric_names = (
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "passage_recall_at_k",
        "passage_coverage_at_k",
        "complete_evidence",
        "fact_coverage",
        "citation_validity",
        "citation_relevance",
        "citation_support",
        "evidence_availability",
    )
    summary = {}
    for name in metric_names:
        values = [record["metrics"][name] for record in records if record.get("metrics", {}).get(name) is not None]
        if values:
            summary[name] = statistics.mean(values)
    passage_records = [
        record
        for record in records
        if record.get("status") == "success" and "passage_recall_at_k" in record.get("metrics", {})
    ]
    passage_values = [
        record["metrics"]["passage_recall_at_k"]
        for record in passage_records
        if record["metrics"].get("passage_recall_at_k") is not None
    ]
    coverage_values = [
        record["metrics"]["passage_coverage_at_k"]
        for record in passage_records
        if record["metrics"].get("passage_coverage_at_k") is not None
    ]
    complete_values = [
        record["metrics"]["complete_passage_evidence"]
        for record in passage_records
        if record["metrics"].get("complete_passage_evidence") is not None
    ]
    summary["passage_examples_evaluated"] = len(passage_values)
    summary["passage_examples_unavailable"] = len(passage_records) - len(passage_values)
    summary["passage_touch_recall_mean"] = statistics.mean(passage_values) if passage_values else None
    summary["weighted_passage_coverage_mean"] = statistics.mean(coverage_values) if coverage_values else None
    summary["complete_passage_evidence_rate"] = statistics.mean(complete_values) if complete_values else None
    successful = [record for record in records if record.get("status") == "success"]
    summary["successful_count"] = len(successful)
    summary["failed_count"] = sum(record.get("status") == "failed" for record in records)
    abstention_records = [record for record in successful if "abstention" in record.get("metrics", {})]
    unanswerable = [record for record in abstention_records if not record["answerable"]]
    answerable = [record for record in abstention_records if record["answerable"]]
    summary["unanswerable_evaluated_count"] = len(unanswerable)
    summary["answerable_evaluated_count"] = len(answerable)
    summary["correct_abstention_count"] = sum(
        record["metrics"]["abstention"] == "correct_abstention" for record in unanswerable
    )
    summary["unsupported_answer_count"] = sum(
        record["metrics"]["abstention"] == "unsupported_answer" for record in unanswerable
    )
    summary["incorrect_refusal_count"] = sum(
        record["metrics"]["abstention"] == "incorrect_refusal" for record in answerable
    )
    if unanswerable:
        summary["correct_abstention_rate"] = summary["correct_abstention_count"] / len(unanswerable)
        summary["unsupported_answer_rate"] = summary["unsupported_answer_count"] / len(unanswerable)
    else:
        summary["correct_abstention_rate"] = None
        summary["unsupported_answer_rate"] = None
    if answerable:
        summary["incorrect_refusal_rate"] = summary["incorrect_refusal_count"] / len(answerable)
    else:
        summary["incorrect_refusal_rate"] = None
    end_to_end_latencies = []
    has_end_to_end_records = any(
        record.get("status") == "success" and ("answer" in record or "pipeline_seconds" in record.get("timing", {}))
        for record in records
    )
    for record in successful:
        timing = record.get("timing", {})
        if not has_end_to_end_records:
            continue
        if "pipeline_seconds" in timing:
            end_to_end_latencies.append(timing["pipeline_seconds"])
        elif "retrieval_seconds" in timing:
            end_to_end_latencies.append(timing["retrieval_seconds"] + timing.get("generation_seconds", 0))
    if latencies:
        summary["retrieval_latency_p50_seconds"] = statistics.median(latencies)
        summary["retrieval_latency_p95_seconds"] = _percentile(latencies, 0.95)
    if end_to_end_latencies:
        summary["end_to_end_latency_p50_seconds"] = statistics.median(end_to_end_latencies)
        summary["end_to_end_latency_p95_seconds"] = _percentile(end_to_end_latencies, 0.95)
    generation_latencies = [
        record["timing"]["generation_seconds"]
        for record in successful
        if "generation_seconds" in record.get("timing", {})
    ]
    if generation_latencies:
        summary["generation_latency_p50_seconds"] = statistics.median(generation_latencies)
        summary["generation_latency_p95_seconds"] = _percentile(generation_latencies, 0.95)
    return summary


def _load_checkpoint_records(checkpoint_dir: Path | None) -> list[dict[str, Any]]:
    if checkpoint_dir is None:
        return []
    records_path = checkpoint_dir / "records.jsonl"
    if not records_path.exists():
        return []
    records_by_id: dict[str, dict[str, Any]] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt checkpoint record in {records_path}") from exc
        if isinstance(value, dict) and value.get("question_id"):
            question_id = str(value["question_id"])
            if question_id in records_by_id:
                raise ValueError(f"Duplicate checkpoint record for question {question_id}")
            records_by_id[question_id] = value
    return list(records_by_id.values())


def _persist_checkpoint(
    checkpoint_dir: Path | None,
    records: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records_path = checkpoint_dir / "records.jsonl"
    temporary = records_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(records_path)
    manifest = checkpoint_dir / "manifest.json"
    manifest_data = dict(metadata or {})
    manifest_data.update(
        {
            "run_status": "running",
            "attempted_count": len(records),
            "successful_count": sum(record.get("status") == "success" for record in records),
            "failed_count": sum(record.get("status") == "failed" for record in records),
        }
    )
    manifest_tmp = manifest.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest_data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest)


def _json_metric(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def experiment_metadata(
    dataset_path: Path,
    corpus_dir: Path,
    split: str,
    top_k: int,
    configuration: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    namespace: str = RAG_VECTOR_NAMESPACE,
) -> dict[str, Any]:
    return {
        "benchmark_version": "1",
        "dataset_name": dataset_path.name,
        "dataset_revision": "f2428dce66ea418a785998fa6ec57bc561b72239" if dataset_path.is_dir() else None,
        "configuration": configuration,
        "dataset_checksum": _sha256(dataset_path),
        "corpus_checksum": _directory_checksum(corpus_dir),
        "split": split,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunker_version": "structural-v2",
        "index_namespace": namespace,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _configuration_snapshot(args: argparse.Namespace, settings: Any | None = None) -> dict[str, Any]:
    integer_fields = {
        "hybrid_dense_k",
        "hybrid_dense_fetch_k",
        "hybrid_sparse_k",
        "hybrid_candidate_k",
        "hybrid_final_k",
    }
    float_fields = {"generation_temperature", "hybrid_dense_weight", "hybrid_sparse_weight"}

    def value(attribute: str, environment: str, default: Any = None) -> Any:
        if settings is not None and hasattr(settings, attribute):
            raw = getattr(settings, attribute)
        elif getattr(args, attribute, None) is not None:
            raw = getattr(args, attribute)
        else:
            raw = os.getenv(environment, default)
        if raw is None:
            return None
        try:
            if attribute in integer_fields:
                return int(raw)
            if attribute in float_fields:
                return float(raw)
        except (TypeError, ValueError):
            return str(raw)
        return raw

    effective_reranking_mode = (
        "true"
        if args.configuration == "hybrid-rerank"
        else "false"
        if args.configuration in {"dense", "hybrid"}
        else value("reranking_mode", "RERANKING_MODE", "auto")
    )
    snapshot = {
        "dataset_name": args.dataset.name,
        "dataset_revision": "f2428dce66ea418a785998fa6ec57bc561b72239" if args.dataset.is_dir() else None,
        "dataset_checksum": _sha256(args.dataset),
        "corpus_checksum": _directory_checksum(args.corpus),
        "split": args.split,
        "configuration": args.configuration,
        "experiment_mode": "end-to-end" if args.end_to_end else "retrieval-only",
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "chunker_version": "structural-v2",
        "llm_provider": value("llm_provider", "LLM_PROVIDER", "ollama"),
        "generation_model": value("generation_model", "GENERATION_MODEL", "llama3.2:3b"),
        "generation_temperature": value("generation_temperature", "GENERATION_TEMPERATURE", "0.7"),
        "generation_output_limit": int(os.getenv("GENERATION_MAX_TOKENS", "2048"))
        if value("llm_provider", "LLM_PROVIDER", "ollama") == "ollama"
        else os.getenv("GENERATION_MAX_TOKENS"),
        "generation_context_limit": os.getenv("GENERATION_CONTEXT_LIMIT"),
        "embedding_provider": None
        if args.configuration in {"standalone", "bm25"}
        else value("embedding_provider", "EMBEDDING_PROVIDER", "ollama"),
        "embedding_model": None
        if args.configuration in {"standalone", "bm25"}
        else value("embedding_model", "EMBEDDING_MODEL", "nomic-embed-text"),
        "vector_store_provider": None
        if args.configuration in {"standalone", "bm25"}
        else value("vector_store_provider", "VECTOR_STORE_PROVIDER", "chroma"),
        "reranker_model": value("reranker_model", "RERANKER_MODEL", "BAAI/bge-reranker-base")
        if args.configuration in {"hybrid-rerank", "full"}
        else None,
        "reranker_device": value("reranker_device", "RERANKER_DEVICE", "cpu")
        if args.configuration in {"hybrid-rerank", "full"}
        else None,
        "reranking_mode": effective_reranking_mode,
        "dense_retrieval_limits": {
            "hybrid_dense_k": value("hybrid_dense_k", "HYBRID_DENSE_K", 10),
            "hybrid_dense_fetch_k": value("hybrid_dense_fetch_k", "HYBRID_DENSE_FETCH_K", 20),
        },
        "sparse_retrieval_limits": {"hybrid_sparse_k": value("hybrid_sparse_k", "HYBRID_SPARSE_K", 10)},
        "fusion_weights": {
            "dense": value("hybrid_dense_weight", "HYBRID_DENSE_WEIGHT", 0.5),
            "sparse": value("hybrid_sparse_weight", "HYBRID_SPARSE_WEIGHT", 0.5),
        },
        "retrieval_policy": {
            "candidate_k": value("hybrid_candidate_k", "HYBRID_CANDIDATE_K", 10),
            "final_k": value("hybrid_final_k", "HYBRID_FINAL_K", 5),
        },
        "evidence_coverage_threshold": float(os.getenv("PASSAGE_COVERAGE_THRESHOLD", "0.8")),
        "prompt_fingerprints": {
            str(path): _sha256(path)
            for path in (
                Path("src/local_rag_chatbot/resources/prompts/master_prompt.json"),
                Path("src/local_rag_chatbot/resources/prompts/tool_calling.json"),
            )
            if path.exists()
        },
    }
    return snapshot


def _configuration_fingerprint(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _attach_configuration_metadata(
    metadata: dict[str, Any], args: argparse.Namespace, settings: Any | None = None
) -> None:
    snapshot = _configuration_snapshot(args, settings)
    metadata["configuration_snapshot"] = snapshot
    metadata["configuration_fingerprint"] = _configuration_fingerprint(snapshot)


def _sha256(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(child.relative_to(path).as_posix().encode())
                digest.update(child.read_bytes())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_checksum(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_run_output(args: argparse.Namespace) -> None:
    args.configuration_snapshot = _configuration_snapshot(args)
    args.configuration_fingerprint = _configuration_fingerprint(args.configuration_snapshot)
    if args.output.suffix == ".json":
        if args.resume:
            raise ValueError("--resume requires a run directory output, not a legacy JSON file.")
        if not args.resume and args.output.exists():
            raise ValueError(f"Refusing to overwrite existing result: {args.output}")
        args.checkpoint_dir = None
        return
    if args.resume:
        if not (args.output / "manifest.json").exists():
            raise ValueError("--resume requires an existing run directory containing manifest.json.")
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_status") in {"completed", "completed_with_failures"}:
            raise ValueError("The run is already completed; refusing to resume or overwrite it.")
        if manifest.get("configuration_fingerprint") != args.configuration_fingerprint:
            raise ValueError(
                "Resume configuration mismatch: effective model, retrieval, dataset, prompt, or runtime settings differ. "
                f"existing={manifest.get('configuration_fingerprint')!r}, requested={args.configuration_fingerprint!r}"
            )
        _load_checkpoint_records(args.output)
        args.checkpoint_dir = args.output
        return
    mode = "end-to-end" if args.end_to_end else "retrieval-only"
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    args.output = args.output / args.configuration / mode / f"run-{run_id}"
    args.checkpoint_dir = args.output


def _run_live(args: argparse.Namespace, examples: list[BenchmarkExample]) -> dict[str, Any]:
    from local_rag_chatbot.providers.llm import create_chat_model
    from local_rag_chatbot.settings import get_settings

    settings = get_settings()
    overrides: dict[str, Any] = {
        "documents_dir": args.corpus,
        "data_dir": args.data_dir,
        "chroma_db_dir": args.data_dir / "chroma",
        "auto_index_on_startup": True,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    for name in (
        "llm_provider",
        "embedding_provider",
        "generation_model",
        "embedding_model",
        "vector_store_provider",
        "reranker_device",
    ):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.reranker_model is not None:
        overrides["reranker_model"] = args.reranker_model
    if args.configuration == "hybrid-rerank":
        overrides["reranking_mode"] = "true"
    elif args.configuration in {"dense", "hybrid"}:
        overrides["reranking_mode"] = "false"
    settings = settings.with_overrides(**overrides)

    if args.configuration == "bm25":
        documents = load_corpus(args.corpus, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        retriever = BM25Retriever(documents)
        metadata = experiment_metadata(
            args.dataset,
            args.corpus,
            args.split,
            args.top_k,
            args.configuration,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        metadata.update(
            {
                "provider": settings.llm_provider,
                "generation_model": settings.generation_model if args.end_to_end else None,
                "embedding_provider": None,
                "embedding_model": None,
                "vector_store_provider": None,
                "retrieval_configuration": "bm25",
            }
        )
        _attach_configuration_metadata(metadata, args, settings)
        if not args.end_to_end:
            return run_retrieval(
                examples,
                retriever,
                configuration=args.configuration,
                split=args.split,
                top_k=args.top_k,
                metadata=metadata,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                retry_failed=args.retry_failed,
            )

        from local_rag_chatbot.documents.prompt_repository import MasterPromptRepository
        from local_rag_chatbot.documents.tool_config_repository import ToolCallingConfigRepository
        from local_rag_chatbot.rag.answer_service import AnswerService
        from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicyResolver

        chat_model = create_chat_model(settings)
        tool_config = ToolCallingConfigRepository(settings.tool_calling_config_file).load()
        answer_service = AnswerService(
            chat_model=chat_model,
            generation_model=settings.generation_model,
            prompt_loader=MasterPromptRepository(settings.master_prompt_file).load,
            answer_prompt_template=tool_config.answer_prompt_template,
            system_prompt_template=tool_config.system_prompt_template,
        )
        return run_end_to_end(
            examples,
            retriever,
            AnswerServiceAdapter(
                answer_service,
                RetrievalPolicyResolver(
                    default_candidate_k=settings.hybrid_candidate_k,
                    default_final_k=settings.hybrid_final_k,
                ),
            ),
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )

    if args.configuration == "standalone":
        chat_model = create_chat_model(settings)
        metadata = experiment_metadata(
            args.dataset,
            args.corpus,
            args.split,
            args.top_k,
            args.configuration,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        metadata.update(
            {
                "provider": settings.llm_provider,
                "generation_model": settings.generation_model,
                "embedding_provider": None,
                "embedding_model": None,
                "vector_store_provider": None,
            }
        )
        _attach_configuration_metadata(metadata, args, settings)
        return run_end_to_end(
            examples,
            None,
            ChatModelAnswerer(chat_model, settings.generation_model),
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )

    from local_rag_chatbot.application import create_application_services, create_indexing_services
    from local_rag_chatbot.providers.embeddings import create_embeddings
    from local_rag_chatbot.rag.retrieval_policy import RetrievalPolicyResolver
    from local_rag_chatbot.engine import ChatbotEngine

    initialization_started = time.perf_counter()
    if args.configuration == "full":
        engine = ChatbotEngine(config=settings)
        event_loop = asyncio.new_event_loop()
        try:
            event_loop.run_until_complete(engine.initialize())
            initialization_seconds = time.perf_counter() - initialization_started
            metadata = experiment_metadata(
                args.dataset,
                args.corpus,
                args.split,
                args.top_k,
                args.configuration,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            metadata.update(
                {
                    "provider": settings.llm_provider,
                    "generation_model": settings.generation_model,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "vector_store_provider": settings.vector_store_provider,
                    "reranker_device": settings.reranker_device,
                    "reranker_model": settings.reranker_model,
                    "initialization_seconds": initialization_seconds,
                    "timing_note": "full mode records total engine query time as pipeline_seconds; component timings are not exposed by the public engine API",
                }
            )
            _attach_configuration_metadata(metadata, args, settings)
            result = run_end_to_end(
                examples,
                None,
                EngineAnswerer(engine, event_loop),
                configuration=args.configuration,
                split=args.split,
                top_k=args.top_k,
                metadata=metadata,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                retry_failed=args.retry_failed,
            )
        finally:
            event_loop.run_until_complete(engine.shutdown())
            event_loop.close()
        result["metadata"] = metadata
        return result

    embeddings = create_embeddings(settings)
    indexing = create_indexing_services(settings, embeddings=embeddings)
    if not indexing.index_manager.prepare():
        raise RuntimeError("The benchmark corpus could not be indexed.")
    initialization_seconds = time.perf_counter() - initialization_started
    documents = indexing.index_manager.documents

    metadata = experiment_metadata(
        args.dataset,
        args.corpus,
        args.split,
        args.top_k,
        args.configuration,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    metadata.update(
        {
            "provider": settings.llm_provider,
            "generation_model": settings.generation_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "vector_store_provider": settings.vector_store_provider,
            "reranker_model": settings.reranker_model if args.configuration == "hybrid-rerank" else None,
            "initialization_seconds": initialization_seconds,
            "retrieval_limits": {
                "dense_k": settings.hybrid_dense_k,
                "dense_fetch_k": settings.hybrid_dense_fetch_k,
                "sparse_k": settings.hybrid_sparse_k,
                "candidate_k": settings.hybrid_candidate_k,
                "final_k": settings.hybrid_final_k,
            },
        }
    )
    _attach_configuration_metadata(metadata, args, settings)

    retriever = build_retriever(
        args.configuration,
        documents,
        vector_backend=indexing.index_manager.vector_backend,
        tuning=indexing.index_manager.retrieval_tuning,
        namespace=RAG_VECTOR_NAMESPACE,
        reranker_model=settings.reranker_model,
        reranker_device=settings.reranker_device,
    )
    if isinstance(retriever, ExistingRetrieverAdapter):
        metadata["reranking_status"] = retriever.retriever.reranking_status
    if args.end_to_end:
        chat_model = create_chat_model(settings)
        services = create_application_services(
            settings,
            chat_model=chat_model,
            embeddings=embeddings,
            vector_backend=indexing.index_manager.vector_backend,
        )
        answerer: Answerer = AnswerServiceAdapter(
            services.answer_service,
            RetrievalPolicyResolver(
                default_candidate_k=settings.hybrid_candidate_k,
                default_final_k=settings.hybrid_final_k,
            ),
        )
        result = run_end_to_end(
            examples,
            retriever,
            answerer,
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    else:
        result = run_retrieval(
            examples,
            retriever,
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    result["metadata"] = metadata
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible RAG benchmark.")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="bm25")
    parser.add_argument("--split", choices=("dev", "test"), default=os.getenv("BENCHMARK_SPLIT", "dev"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("BENCHMARK_TOP_K", "5")))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "500")))
    parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "100")))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/datapizza"))
    parser.add_argument("--corpus", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark.json"))
    parser.add_argument("--resume", action="store_true", help="Resume an existing run directory.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed questions while resuming.")
    parser.add_argument("--run-id", help="Explicit run identifier for a new directory output.")
    parser.add_argument("--live", action="store_true", help="Initialize configured providers and run live evaluation.")
    parser.add_argument(
        "--mode",
        choices=("retrieval-only", "end-to-end"),
        default=None,
        help="Evaluation mode; BENCHMARK_MODE is used when omitted.",
    )
    parser.add_argument("--end-to-end", action="store_true", help="Backward-compatible alias for --mode end-to-end.")
    parser.add_argument("--data-dir", type=Path, default=Path(".benchmark-data"))
    parser.add_argument("--llm-provider")
    parser.add_argument("--generation-model")
    parser.add_argument("--embedding-provider")
    parser.add_argument("--embedding-model")
    parser.add_argument("--vector-store-provider")
    parser.add_argument("--reranker-model")
    parser.add_argument("--reranker-device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    requested_mode = args.mode or os.getenv("BENCHMARK_MODE", "retrieval-only")
    if args.end_to_end and requested_mode == "retrieval-only":
        parser.error("--end-to-end conflicts with --mode retrieval-only")
    args.end_to_end = args.end_to_end or requested_mode == "end-to-end"
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.chunk_size < 1 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-size must be positive and --chunk-overlap must be between 0 and chunk-size - 1")
    try:
        validate_configuration_mode(args.configuration, "end-to-end" if args.end_to_end else "retrieval-only")
    except ValueError as exc:
        parser.error(str(exc))
    if args.configuration == "standalone" and args.live is False:
        parser.error("standalone requires --live")
    if args.configuration == "full" and args.live is False:
        parser.error("full requires --live")
    if args.configuration != "bm25" and not args.live:
        parser.error(
            f"{args.configuration} requires --live so its configured providers and index are initialized explicitly"
        )
    try:
        _prepare_run_output(args)
    except ValueError as exc:
        parser.error(str(exc))
    examples = load_examples(args.dataset, args.split)
    if args.live:
        result = _run_live(args, examples)
    else:
        documents = load_corpus(args.corpus, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        metadata = experiment_metadata(
            args.dataset,
            args.corpus,
            args.split,
            args.top_k,
            args.configuration,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        _attach_configuration_metadata(metadata, args)
        result = run_retrieval(
            examples,
            BM25Retriever(documents),
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
    if args.checkpoint_dir is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest.update(
            {
                "run_status": ("completed_with_failures" if result["summary"].get("failed_count", 0) else "completed"),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "attempted_count": len(result["records"]),
                "successful_count": result["summary"].get("successful_count", 0),
                "failed_count": result["summary"].get("failed_count", 0),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "summary.json").write_text(
            json.dumps(result["summary"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
