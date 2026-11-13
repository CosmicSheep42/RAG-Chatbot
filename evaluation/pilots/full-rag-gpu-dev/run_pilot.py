"""Run exactly the fixed Datapizza development split through ChatbotEngine."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import os
import json
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
from typing import Any

from evaluation.adapters.datapizza import load_examples
from local_rag_chatbot.engine import create_default_engine
from local_rag_chatbot.settings import get_settings


ROOT = Path(__file__).resolve().parents[3]
SPLITS = ROOT / "evaluation/dataset/datapizza/splits.json"
RUN_ROOT = Path(os.getenv("PILOT_OUTPUT_ROOT", ROOT / "evaluation/pilots/full-rag-gpu-dev"))
RUN_ID = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
RUN_DIR = RUN_ROOT / RUN_ID
DOC_KEYS = (
    "source",
    "source_name",
    "context_role",
    "header_path",
    "hierarchy_path",
    "section",
    "structural_type",
    "chunk_index",
    "document_id",
    "chunk_id",
    "source_char_start",
    "source_char_end",
    "retrieval_dense_score",
    "retrieval_dense_rank",
    "retrieval_sparse_score",
    "retrieval_sparse_rank",
    "retrieval_reranker_score",
    "retrieval_final_ranking_score",
    "retrieval_score",
)
REFUSAL_PATTERN = re.compile(
    r"^\s*(?:I\s+(?:do not have enough|don't have enough|cannot answer|can't answer|"
    r"am unable to answer|cannot provide information|can't provide information|cannot help|can't help)|"
    r"I(?:'m| am) unable to assist)\b",
    re.I,
)


def _document(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    return {
        **{key: metadata[key] for key in DOC_KEYS if key in metadata},
        "content": str(getattr(document, "page_content", "")),
    }


def _key(document: dict[str, Any]) -> tuple[Any, ...]:
    return document.get("source", "unknown"), document.get("content", "")


def _matches_gold(document: dict[str, Any], passages: tuple[Any, ...]) -> bool:
    return any(
        document.get("source") == passage.document_path
        and document.get("source_char_start", 0) < passage.end_char
        and document.get("source_char_end", 0) > passage.start_char
        for passage in passages
    )


def _model(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _git_revision() -> str:
    if revision := os.getenv("PILOT_GIT_REVISION"):
        return revision
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
    except FileNotFoundError:
        git_dir = ROOT / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            ref_path = git_dir / ref
            if ref_path.is_file():
                return ref_path.read_text(encoding="utf-8").strip()
            for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split()[0]
        return head


def _coverage(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "sufficient": value.sufficient,
        "score": value.score,
        "reason": value.reason,
        "dominant_source": value.dominant_source,
        "selected_sources": value.selected_sources,
        "selected_chunks": value.selected_chunks,
        "total_available_chunks": value.total_available_chunks,
        "selected_units": value.selected_units,
        "total_available_units": value.total_available_units,
    }


def _latencies(records: list[dict[str, Any]]) -> dict[str, float | None]:
    values = sorted(r["latency_seconds"]["total"] for r in records if r.get("status") == "completed")
    if not values:
        return {"p50": None, "p95": None, "max": None}

    def percentile(percent: float) -> float:
        position = (len(values) - 1) * percent
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"p50": percentile(0.50), "p95": percentile(0.95), "max": values[-1]}


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in records if r.get("status") == "completed"]
    calls = [call for r in done for call in r.get("recovery", {}).get("tool_events", [])]
    search_calls = [c for c in calls if c.get("name") == "search_documents"]
    citation_count = sum(len(r.get("generation", {}).get("citation_identifiers", [])) for r in done)
    invalid_citation_count = sum(len(r.get("generation", {}).get("invalid_citation_identifiers", [])) for r in done)
    recovery_questions = sum(bool(r.get("recovery", {}).get("attempted")) for r in done)
    return {
        "expected_questions": 12,
        "completed": len(done),
        "failed_executions": 12 - len(done),
        "explicit_refusals": sum(bool(r.get("generation", {}).get("explicit_refusal")) for r in done),
        "textual_refusal_heuristic": sum(bool(r.get("generation", {}).get("textual_refusal_heuristic")) for r in done),
        "tool_calls": len(calls),
        "valid_tool_calls": sum(c.get("payload", {}).get("status") in {"success", "success_empty"} for c in calls),
        "tool_validation_failures": sum(c.get("payload", {}).get("status") == "validation_error" for c in calls),
        "tool_execution_failures": sum(c.get("payload", {}).get("status") == "execution_error" for c in calls),
        "empty_searches": sum(c.get("payload", {}).get("status") == "success_empty" for c in search_calls),
        "searches_returning_documents": sum(c.get("payload", {}).get("status") == "success" for c in search_calls),
        "duplicate_only_searches": sum(
            c.get("payload", {}).get("status") == "success"
            and c.get("result_count", 0) > 0
            and c.get("new_document_count", 0) == 0
            for c in search_calls
        ),
        "new_recovered_documents_selected_in_final_context": sum(
            c.get("new_evidence_selected_count", 0) for c in search_calls
        ),
        "questions_with_new_recovered_evidence_selected": sum(
            any(c.get("new_evidence_selected_count", 0) > 0 for c in r.get("recovery", {}).get("tool_events", []))
            for r in done
        ),
        "questions_recovered_to_answer_decision": sum(
            bool(r.get("recovery", {}).get("recovered_to_answer_decision")) for r in done
        ),
        "recovery_searches_introducing_new_documents": sum(
            c.get("new_document_count", 0) > 0
            for r in done
            if r.get("recovery", {}).get("attempted")
            for c in r.get("recovery", {}).get("tool_events", [])
            if c.get("name") == "search_documents"
        ),
        "successful_recovery_searches": sum(
            c.get("new_gold_overlap_document_count", 0) > 0
            for r in done
            if r.get("recovery", {}).get("attempted")
            for c in r.get("recovery", {}).get("tool_events", [])
            if c.get("name") == "search_documents"
        ),
        "recovery_searches_with_new_evidence_selected": sum(
            c.get("new_evidence_selected_count", 0) > 0
            for r in done
            if r.get("recovery", {}).get("attempted")
            for c in r.get("recovery", {}).get("tool_events", [])
            if c.get("name") == "search_documents"
        ),
        "questions_with_gold_overlap_recovery": sum(
            any(c.get("new_gold_overlap_document_count", 0) > 0 for c in r.get("recovery", {}).get("tool_events", []))
            for r in done
            if r.get("recovery", {}).get("attempted")
        ),
        "additional_evidence_documents": sum(r.get("recovery", {}).get("new_document_count", 0) for r in done),
        "questions_requiring_recovery": recovery_questions,
        "questions_requiring_recovery_percent": recovery_questions / 12 * 100,
        "tool_calls_per_question": len(calls) / len(done) if done else None,
        "citation_identifiers": citation_count,
        "invalid_citation_identifiers": invalid_citation_count,
        "valid_citation_identifier_rate": 1 - invalid_citation_count / citation_count if citation_count else None,
        "incomplete_responses": sum(bool(r.get("generation", {}).get("partial_answer")) for r in done),
        "latency_seconds": _latencies(done),
    }


async def main() -> None:
    started_at = datetime.now(UTC)
    split_data = json.loads(SPLITS.read_text(encoding="utf-8"))
    examples = load_examples(ROOT / "evaluation/dataset/datapizza", "dev")
    expected_ids = [example.id for example in examples]
    if set(expected_ids) != set(split_data["dev"]) or len(set(expected_ids)) != 12:
        raise RuntimeError("Development examples do not exactly match the 12 IDs in splits.json.")

    settings = get_settings().with_overrides(
        documents_dir=ROOT / "knowledge-base",
        data_dir=Path("/app/.benchmark-data"),
        chroma_db_dir=Path("/app/.benchmark-data/chroma"),
        router_store_dir=Path("/app/.benchmark-data/router_data"),
        generation_model="llama3.2:3b",
        embedding_model="nomic-embed-text",
        llm_provider="ollama",
        embedding_provider="ollama",
        vector_store_provider="chroma",
        generation_temperature=0,
        chunk_size=500,
        chunk_overlap=100,
        hybrid_final_k=5,
        reranker_device="cpu",
        max_concurrent_inferences=1,
        auto_index_on_startup=False,
    )
    engine = create_default_engine(settings)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    await engine.initialize()
    chatbot = engine.services.chatbot
    trace_lock = threading.Lock()
    current: dict[str, Any] = {}

    retrieval_service = chatbot.index_manager.retrieval_service
    original_retrieve = retrieval_service.retrieve

    def capture_retrieve(query: str, *args: Any, **kwargs: Any) -> Any:
        before = time.perf_counter()
        result = original_retrieve(query, *args, **kwargs)
        event = {
            "query": query,
            "latency_seconds": time.perf_counter() - before,
            "candidate_documents": [_document(d) for d in result.candidates],
            "retrieved_documents": [_document(d) for d in result.documents],
            "confidence": result.confidence.as_debug_payload() if result.confidence else None,
            "coverage": _coverage(result.coverage),
            "policy": {
                "intent": result.policy.intent.value if result.policy else None,
                "retrieval_limit": result.policy.retrieval_limit if result.policy else None,
                "context_limit": result.policy.context_limit if result.policy else None,
            },
            "metadata": dict(result.metadata),
        }
        with trace_lock:
            current.setdefault("retrieval_events", []).append(event)
        return result

    retrieval_service.retrieve = capture_retrieve
    original_select = chatbot._select_and_assess_final_context

    def capture_select(query: str, documents: list[Any], policy: Any) -> Any:
        selected, coverage, assessment = original_select(query, documents, policy)
        with trace_lock:
            current.setdefault("context_selections", []).append(
                {
                    "input_documents": [_document(d) for d in documents],
                    "selected_documents": [_document(d) for d in selected],
                    "coverage": _coverage(coverage),
                    "confidence": assessment.as_debug_payload() if assessment else None,
                    "available_evidence_references": [
                        citation["citation_id"] for citation in chatbot.context_builder.build_citations(selected)
                    ],
                }
            )
        return selected, coverage, assessment

    chatbot._select_and_assess_final_context = capture_select
    original_optional_tools = chatbot._agenerate_answer_with_optional_tools

    async def capture_optional_tools(*args: Any, **kwargs: Any) -> Any:
        current["recovery_phase_invoked"] = bool(kwargs.get("recovery_only"))
        before = time.perf_counter()
        try:
            return await original_optional_tools(*args, **kwargs)
        finally:
            current["tool_phase_seconds"] = time.perf_counter() - before

    chatbot._agenerate_answer_with_optional_tools = capture_optional_tools
    original_execute = chatbot.tool_registry.execute

    def capture_execute(name: str, arguments: dict[str, Any], *, request_context: Any = None) -> Any:
        before = time.perf_counter()
        try:
            content, documents, payload = original_execute(name, arguments, request_context=request_context)
        except Exception as exc:
            with trace_lock:
                current.setdefault("tool_events", []).append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "payload": {"status": "raised_error", "error": type(exc).__name__},
                        "documents": [],
                        "latency_seconds": time.perf_counter() - before,
                    }
                )
            raise
        with trace_lock:
            current.setdefault("tool_events", []).append(
                {
                    "name": name,
                    "arguments": arguments,
                    "payload": payload,
                    "documents": [_document(d) for d in documents],
                    "latency_seconds": time.perf_counter() - before,
                }
            )
        return content, documents, payload

    chatbot.tool_registry.execute = capture_execute
    original_generation = chatbot.answer_service.agenerate

    async def capture_generation(*args: Any, **kwargs: Any) -> Any:
        before = time.perf_counter()
        try:
            return await original_generation(*args, **kwargs)
        finally:
            current["generation_seconds"] = current.get("generation_seconds", 0.0) + time.perf_counter() - before

    chatbot.answer_service.agenerate = capture_generation
    original_answer = chatbot.achat

    async def capture_answer(*args: Any, **kwargs: Any) -> Any:
        result = await original_answer(*args, **kwargs)
        current["answer_trace"] = result
        return result

    chatbot.achat = capture_answer

    records: list[dict[str, Any]] = []
    try:
        for example in examples:
            current.clear()
            total_start = time.perf_counter()
            try:
                response = await engine.chat(example.question)
                total_latency = time.perf_counter() - total_start
                trace = current.get("answer_trace", {})
                selection_events = current.get("context_selections", [])
                initial_selection = selection_events[0] if selection_events else None
                final_selection = selection_events[-1] if selection_events else None
                initial_docs = initial_selection["input_documents"] if initial_selection else []
                selected_docs = final_selection["selected_documents"] if final_selection else []
                initial_candidates = (
                    current.get("retrieval_events", [{}])[0].get("candidate_documents", [])
                    if current.get("retrieval_events")
                    else []
                )
                captured_tool_events = list(current.get("tool_events", []))
                tool_events = []
                for trace_event in trace.get("tool_calls", []):
                    captured_index = next(
                        (
                            index
                            for index, event in enumerate(captured_tool_events)
                            if event.get("name") == trace_event.get("name")
                            and event.get("arguments") == trace_event.get("arguments")
                        ),
                        None,
                    )
                    captured = captured_tool_events.pop(captured_index) if captured_index is not None else {}
                    event = {**trace_event, **captured, "id": trace_event.get("id")}
                    event["origin"] = "fallback" if event["id"] == "recovery-fallback" else "model"
                    tool_events.append(event)
                tool_events.extend(captured_tool_events)
                search_events = [e for e in tool_events if e.get("name") == "search_documents"]
                initial_keys = {_key(d) for d in initial_docs}
                selected_keys = {_key(d) for d in selected_docs}
                known_keys = set(initial_keys)
                recovered = []
                for event in search_events:
                    added_documents = [d for d in event.get("documents", []) if _key(d) not in known_keys]
                    event["new_document_count"] = len(added_documents)
                    event["new_gold_overlap_document_count"] = sum(
                        _matches_gold(document, example.passages) for document in added_documents
                    )
                    event["new_evidence_selected_count"] = sum(
                        _key(document) in selected_keys for document in added_documents
                    )
                    recovered.extend(added_documents)
                    known_keys.update(_key(document) for document in added_documents)
                omitted = [d for d in initial_docs if _key(d) not in selected_keys]
                initial_decision = (
                    (initial_selection or {}).get("confidence", {}).get("decision") if initial_selection else None
                )
                final_decision = (
                    (final_selection or {}).get("confidence", {}).get("decision") if final_selection else None
                )
                recovered_to_answer = bool(
                    initial_decision != "answer"
                    and final_decision == "answer"
                    and any(event.get("new_evidence_selected_count", 0) for event in search_events)
                )
                citation_ids = [c.citation_id for c in response.citations]
                mentioned = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", response.result)))
                invalid = [identifier for identifier in mentioned if identifier not in citation_ids]
                answer = response.result
                records.append(
                    {
                        "question_id": example.id,
                        "split": "dev",
                        "question": example.question,
                        "reference_answer": example.answer,
                        "gold_passages": [passage.__dict__ for passage in example.passages],
                        "status": "completed",
                        "retrieval": {
                            "query": trace.get("standalone_query"),
                            "query_rewritten": trace.get("query_rewritten"),
                            "attempts": current.get("retrieval_events", []),
                            "candidate_document_count": len(initial_candidates),
                            "retrieved_source_ids": list(
                                dict.fromkeys(d.get("source", d.get("document_id")) for d in initial_docs)
                            ),
                            "confidence": initial_selection.get("confidence")
                            if initial_selection
                            else trace.get("retrieval_confidence_assessment"),
                            "coverage": initial_selection.get("coverage") if initial_selection else None,
                        },
                        "context": {
                            "candidate_document_count": len(initial_docs),
                            "selected_document_count": len(selected_docs),
                            "selected_source_ids": list(
                                dict.fromkeys(d.get("source", d.get("document_id")) for d in selected_docs)
                            ),
                            "omitted_document_count": len(omitted),
                            "context_selection_events": selection_events,
                            "relevant_evidence_removed": any(
                                any(
                                    d.get("source") == passage.document_path
                                    and d.get("source_char_start", 0) < passage.end_char
                                    and d.get("source_char_end", 0) > passage.start_char
                                    for d in omitted
                                )
                                for passage in example.passages
                            ),
                        },
                        "recovery": {
                            "attempted": current.get("recovery_phase_invoked", False),
                            "attempt_count": len(search_events) if current.get("recovery_phase_invoked") else 0,
                            "tool_call_count": len(tool_events),
                            "validation_failure_count": sum(
                                e.get("payload", {}).get("status") == "validation_error" for e in tool_events
                            ),
                            "execution_failure_count": sum(
                                e.get("payload", {}).get("status") in {"execution_error", "raised_error"}
                                for e in tool_events
                            ),
                            "empty_search_count": sum(
                                e.get("payload", {}).get("status") == "success_empty" for e in search_events
                            ),
                            "document_search_count": sum(
                                e.get("payload", {}).get("status") == "success" for e in search_events
                            ),
                            "new_document_count": len(recovered) if current.get("recovery_phase_invoked") else 0,
                            "new_gold_overlap_document_count": sum(
                                e.get("new_gold_overlap_document_count", 0) for e in search_events
                            ),
                            "new_evidence_selected_count": sum(
                                e.get("new_evidence_selected_count", 0) for e in search_events
                            ),
                            "recovered_to_answer_decision": recovered_to_answer,
                            "new_documents": recovered,
                            "tool_events": tool_events,
                        },
                        "generation": {
                            "answer": answer,
                            "explicit_refusal": bool(trace.get("refused")),
                            "textual_refusal_heuristic": bool(REFUSAL_PATTERN.search(answer)),
                            "partial_answer": bool(trace.get("partial_context")),
                            "citation_identifiers": mentioned,
                            "available_citation_identifiers": citation_ids,
                            "invalid_citation_identifiers": invalid,
                            "citations": [_model(c) for c in response.citations],
                            "sources": [_model(s) for s in response.sources],
                            "available_evidence_references": (
                                final_selection.get("available_evidence_references", []) if final_selection else []
                            ),
                            "assessment": "Explicitly refused"
                            if trace.get("refused")
                            else "Cannot determine automatically",
                        },
                        "latency_seconds": {
                            "total": total_latency,
                            "retrieval": sum(e["latency_seconds"] for e in current.get("retrieval_events", [])),
                            "generation": current.get("generation_seconds", 0.0),
                            "tool_overhead": sum(e.get("latency_seconds", 0.0) for e in current.get("tool_events", [])),
                            "tool_phase": current.get("tool_phase_seconds", 0.0),
                        },
                        "errors": [],
                    }
                )
            except Exception as exc:
                records.append(
                    {
                        "question_id": example.id,
                        "split": "dev",
                        "question": example.question,
                        "reference_answer": example.answer,
                        "gold_passages": [passage.__dict__ for passage in example.passages],
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_seconds": {"total": time.perf_counter() - total_start},
                    }
                )
            finally:
                with (RUN_DIR / "records.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
    finally:
        await engine.shutdown()

    record_ids = [record["question_id"] for record in records]
    if record_ids != expected_ids or len(set(record_ids)) != 12:
        raise RuntimeError("Pilot records do not contain each expected development question exactly once.")
    persisted_records = [
        json.loads(line)
        for line in (RUN_DIR / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if persisted_records != records:
        raise RuntimeError("Persisted records differ from the records used to calculate summary metrics.")

    finished_at = datetime.now(UTC)
    revision = _git_revision()
    try:
        status_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        tracked_and_untracked = [line[3:] for line in status_lines]
        tree_dirty: bool | None = bool(status_lines)
    except FileNotFoundError:
        injected_paths = os.getenv("PILOT_WORKING_TREE_PATHS_JSON")
        tracked_and_untracked = json.loads(injected_paths) if injected_paths else []
        tree_dirty = bool(tracked_and_untracked) if injected_paths is not None else None
    if fingerprint := os.getenv("PILOT_SOURCE_FINGERPRINT_SHA256"):
        tree_fingerprint = fingerprint
    else:
        tree_hash = hashlib.sha256()
        for relative_path in sorted(tracked_and_untracked):
            path = ROOT / relative_path
            if path.is_file():
                tree_hash.update(relative_path.encode())
                tree_hash.update(path.read_bytes())
        tree_fingerprint = tree_hash.hexdigest()
    versions = {}
    for package in ("local-rag-chatbot", "langchain", "langchain-core", "chromadb", "langchain-ollama", "ollama"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    manifest = {
        "run_id": RUN_ID,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "git_revision": revision,
        "working_tree_dirty": tree_dirty,
        "working_tree_status_available": tree_dirty is not None,
        "working_tree_paths": tracked_and_untracked,
        "working_tree_fingerprint_sha256": tree_fingerprint,
        "python_version": platform.python_version(),
        "runtime_dependencies": versions,
        "gpu": {
            "model": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_mib": 4096,
            "driver": "550.163.01",
            "runtime_confirmation": "see REPORT.md",
        },
        "ollama": {
            "image": "ollama/ollama:0.32.1",
            "generation_model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "reranker": "unavailable; not installed",
        },
        "effective_rag_configuration": {
            "pipeline": "ChatbotEngine Full production RAG",
            "temperature": settings.generation_temperature,
            "chunk_size": 500,
            "chunk_overlap": 100,
            "top_k": 5,
            "max_concurrent_inferences": settings.max_concurrent_inferences,
            "reranker_device": settings.reranker_device,
            "recovery": "current production behavior",
        },
        "dataset": {
            "name": split_data.get("dataset"),
            "revision": split_data.get("source_revision"),
            "manifest_sha256": hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
        },
        "question_ids": expected_ids,
    }
    summary = _metrics(records)
    (RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(RUN_DIR), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
