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
import platform
import statistics
import subprocess
import sys
import time
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
    complete_evidence_retrieval,
    fact_coverage,
    ndcg_at_k,
    passage_recall_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

CONFIGURATIONS = ("standalone", "dense", "bm25", "hybrid", "hybrid-rerank", "full")


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
        loaded = loader.load(path, base_metadata={"source": path.name, "document_id": path.name})
        documents.extend(Document(item.page_content, metadata=item.metadata) for item in loaded)
    chunks = splitter.split(documents)
    cursors: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        source_text = next(
            (document.page_content for document in documents if document.metadata.get("source") == source), ""
        )
        start = source_text.find(chunk.page_content, cursors.get(source, 0))
        if start < 0:
            start = source_text.find(chunk.page_content)
        if start >= 0:
            chunk.metadata["source_char_start"] = start
            chunk.metadata["source_char_end"] = start + len(chunk.page_content)
            cursors[source] = start + 1
    return chunks


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
) -> dict[str, Any]:
    records = []
    retrieval_latencies = []
    for example in examples:
        started = time.perf_counter()
        try:
            documents = retriever.retrieve(example.question, top_k)
        except Exception as exc:
            records.append(
                {
                    "question_id": example.id,
                    "configuration": configuration,
                    "split": split,
                    "answerable": example.answerable,
                    "status": "failed",
                    "error": str(exc),
                    "timing": {"retrieval_seconds": time.perf_counter() - started},
                }
            )
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
        records.append(
            {
                "question_id": example.id,
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
                    "complete_evidence": complete_evidence_retrieval(ranked, example.relevant_documents, top_k),
                },
                "timing": {"retrieval_seconds": latency},
                "status": "success",
            }
        )
    return {
        "metadata": metadata or {},
        "configuration": configuration,
        "split": split,
        "records": records,
        "summary": _summary(records, retrieval_latencies),
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
) -> dict[str, Any]:
    records = []
    retrieval_latencies = []
    for example in examples:
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
                    "configuration": configuration,
                    "split": split,
                    "status": "failed",
                    "error": str(exc),
                    "timing": {"retrieval_seconds": time.perf_counter() - started},
                }
            )
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
    return {
        "metadata": metadata or {},
        "configuration": configuration,
        "split": split,
        "records": records,
        "summary": _summary(records, retrieval_latencies),
    }


def _summary(records: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    metric_names = (
        "recall_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "passage_recall_at_k",
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
    for record in successful:
        timing = record.get("timing", {})
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
    for name in ("llm_provider", "embedding_provider", "generation_model", "embedding_model", "vector_store_provider"):
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
        return run_end_to_end(
            examples,
            None,
            ChatModelAnswerer(chat_model, settings.generation_model),
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
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
                    "reranker_model": settings.reranker_model,
                    "initialization_seconds": initialization_seconds,
                    "timing_note": "full mode records total engine query time as pipeline_seconds; component timings are not exposed by the public engine API",
                }
            )
            result = run_end_to_end(
                examples,
                None,
                EngineAnswerer(engine, event_loop),
                configuration=args.configuration,
                split=args.split,
                top_k=args.top_k,
                metadata=metadata,
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

    retriever = build_retriever(
        args.configuration,
        documents,
        vector_backend=indexing.index_manager.vector_backend,
        tuning=indexing.index_manager.retrieval_tuning,
        namespace=RAG_VECTOR_NAMESPACE,
        reranker_model=settings.reranker_model,
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
        )
    else:
        result = run_retrieval(
            examples,
            retriever,
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=metadata,
        )
    result["metadata"] = metadata
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible RAG benchmark.")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="bm25")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/datapizza"))
    parser.add_argument("--corpus", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark.json"))
    parser.add_argument("--live", action="store_true", help="Initialize configured providers and run live evaluation.")
    parser.add_argument("--end-to-end", action="store_true", help="Generate answers instead of retrieval-only records.")
    parser.add_argument("--data-dir", type=Path, default=Path(".benchmark-data"))
    parser.add_argument("--llm-provider")
    parser.add_argument("--generation-model")
    parser.add_argument("--embedding-provider")
    parser.add_argument("--embedding-model")
    parser.add_argument("--vector-store-provider")
    parser.add_argument("--reranker-model")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.chunk_size < 1 or args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-size must be positive and --chunk-overlap must be between 0 and chunk-size - 1")
    if args.configuration != "bm25" and not args.live:
        parser.error(
            f"{args.configuration} requires --live so its configured providers and index are initialized explicitly"
        )
    examples = load_examples(args.dataset, args.split)
    if args.live:
        result = _run_live(args, examples)
    else:
        documents = load_corpus(args.corpus, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        result = run_retrieval(
            examples,
            BM25Retriever(documents),
            configuration=args.configuration,
            split=args.split,
            top_k=args.top_k,
            metadata=experiment_metadata(
                args.dataset,
                args.corpus,
                args.split,
                args.top_k,
                args.configuration,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
