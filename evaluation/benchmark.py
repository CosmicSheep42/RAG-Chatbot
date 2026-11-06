"""Small, offline-capable benchmark runner for the fixed evaluation corpus.

Live model providers are intentionally injected by callers. Importing this module
never initializes a model, vector store, or network client.
"""

from __future__ import annotations

import argparse
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
from local_rag_chatbot.vectorstores.contracts import VectorStoreBackend

from .metrics import (
    abstention_result,
    citation_quality,
    complete_evidence_retrieval,
    fact_coverage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

CONFIGURATIONS = ("standalone", "dense", "bm25", "hybrid", "hybrid-rerank", "full")


class Retriever(Protocol):
    def retrieve(self, question: str, limit: int) -> list[Document]: ...


class Answerer(Protocol):
    def answer(self, question: str, documents: list[Document]) -> tuple[str, list[str]]: ...


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
    namespace: str = "benchmark",
    tuning: RetrievalTuning | None = None,
) -> Retriever | None:
    """Build an adapter without changing or duplicating production retrieval logic."""
    if configuration == "standalone":
        return None
    if configuration == "bm25":
        return BM25Retriever(documents)
    if vector_backend is None:
        raise ValueError(f"{configuration} requires an injected vector-store backend")
    if configuration == "dense":
        return DenseRetriever(vector_backend, namespace)
    if configuration in {"hybrid", "hybrid-rerank", "full"}:
        return ExistingRetrieverAdapter(
            HybridRetriever(
                vector_backend=vector_backend,
                namespace=namespace,
                documents=documents,
                tuning=tuning or RetrievalTuning(),
                reranking_mode="true" if configuration == "hybrid-rerank" else "false",
            )
        )
    raise ValueError(f"Unknown benchmark configuration: {configuration}")


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    question: str
    answerable: bool
    reference_answer: str | None
    relevant_documents: list[str]
    required_facts: list[str]


def load_corpus(corpus_dir: Path, *, chunk_size: int = 800, chunk_overlap: int = 80) -> list[Document]:
    documents: list[Document] = []
    loader = MarkdownLoader()
    splitter = DocumentSplitter(chunk_size, chunk_overlap)
    for path in sorted(corpus_dir.glob("*.md")):
        loaded = loader.load(path, base_metadata={"source": path.name, "document_id": path.name})
        documents.extend(Document(item.page_content, metadata=item.metadata) for item in loaded)
    return splitter.split(documents)


def load_examples(dataset_path: Path, split: str) -> list[BenchmarkExample]:
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
        documents = retriever.retrieve(example.question, top_k)
        latency = time.perf_counter() - started
        retrieval_latencies.append(latency)
        ranked = [
            {"document_id": str(document.metadata.get("source", "")), "rank": rank}
            for rank, document in enumerate(documents, 1)
        ]
        records.append(
            {
                "question_id": example.id,
                "configuration": configuration,
                "split": split,
                "retrieved_documents": ranked,
                "metrics": {
                    "recall_at_k": _json_metric(recall_at_k(ranked, example.relevant_documents, top_k)),
                    "mrr_at_k": reciprocal_rank_at_k(ranked, example.relevant_documents, top_k),
                    "ndcg_at_k": _json_metric(ndcg_at_k(ranked, example.relevant_documents, top_k)),
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
) -> dict[str, Any]:
    records = []
    for example in examples:
        started = time.perf_counter()
        documents = retriever.retrieve(example.question, top_k) if retriever else []
        retrieval_seconds = time.perf_counter() - started
        generation_started = time.perf_counter()
        answer, citations = answerer.answer(example.question, documents)
        generation_seconds = time.perf_counter() - generation_started
        retrieved = [str(document.metadata.get("source", "")) for document in documents]
        citation_metrics = citation_quality(citations, retrieved, example.relevant_documents)
        records.append(
            {
                "question_id": example.id,
                "configuration": configuration,
                "split": split,
                "retrieved_documents": retrieved,
                "answer": answer,
                "citations": citations,
                "metrics": {
                    "fact_coverage": fact_coverage(answer, example.required_facts) if example.answerable else None,
                    "grounded_fact_coverage": fact_coverage(
                        " ".join(d.page_content for d in documents), example.required_facts
                    ),
                    "citation_validity": citation_metrics["validity"],
                    "citation_relevance": citation_metrics["relevance"],
                    "abstention": abstention_result(answer, example.answerable),
                },
                "timing": {"retrieval_seconds": retrieval_seconds, "generation_seconds": generation_seconds},
                "status": "success",
            }
        )
    return {"configuration": configuration, "split": split, "records": records, "summary": _summary(records, [])}


def _summary(records: list[dict[str, Any]], latencies: list[float]) -> dict[str, Any]:
    metric_names = ("recall_at_k", "mrr_at_k", "ndcg_at_k", "fact_coverage", "citation_validity")
    summary = {}
    for name in metric_names:
        values = [record["metrics"][name] for record in records if record["metrics"].get(name) is not None]
        if values:
            summary[name] = statistics.mean(values)
    if latencies:
        summary["retrieval_latency_p50_seconds"] = statistics.median(latencies)
        summary["retrieval_latency_p95_seconds"] = _percentile(latencies, 0.95)
    return summary


def _json_metric(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def experiment_metadata(
    dataset_path: Path, corpus_dir: Path, split: str, top_k: int, configuration: str
) -> dict[str, Any]:
    return {
        "benchmark_version": "1",
        "configuration": configuration,
        "dataset_checksum": _sha256(dataset_path),
        "corpus_checksum": _directory_checksum(corpus_dir),
        "split": split,
        "top_k": top_k,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _sha256(path: Path) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline retrieval benchmark.")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="bm25")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/questions.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/corpus"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark.json"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.configuration != "bm25":
        parser.error(
            f"{args.configuration} needs an explicit provider/vector-store adapter; the offline CLI only runs bm25"
        )
    examples = load_examples(args.dataset, args.split)
    documents = load_corpus(args.corpus)
    result = run_retrieval(
        examples,
        BM25Retriever(documents),
        configuration=args.configuration,
        split=args.split,
        top_k=args.top_k,
        metadata=experiment_metadata(args.dataset, args.corpus, args.split, args.top_k, args.configuration),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
