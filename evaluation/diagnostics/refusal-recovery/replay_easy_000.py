"""Replay only the already-exposed held-out diagnostic case, never its split peers."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

from local_rag_chatbot.engine import create_default_engine
from local_rag_chatbot.settings import get_settings


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(os.getenv("DIAGNOSTIC_OUTPUT", "/tmp/refusal-diagnostics/easy-000.json"))
QUESTION_ID = "datapizza-easy-000"


def document_payload(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    return {
        "source": metadata.get("source"),
        "chunk_id": metadata.get("chunk_id"),
        "source_char_start": metadata.get("source_char_start"),
        "source_char_end": metadata.get("source_char_end"),
        "dense_score": metadata.get("retrieval_dense_score"),
        "dense_rank": metadata.get("retrieval_dense_rank"),
        "sparse_score": metadata.get("retrieval_sparse_score"),
        "sparse_rank": metadata.get("retrieval_sparse_rank"),
        "content": getattr(document, "page_content", ""),
    }


async def main() -> None:
    item = next(
        row
        for row in json.loads((ROOT / "evaluation/dataset/datapizza/easy.json").read_text(encoding="utf-8"))
        if int(row["id"]) == 0
    )
    question = item["question"]
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
    await engine.initialize()
    chatbot = engine.services.chatbot
    evidence: dict[str, Any] = {"question_id": QUESTION_ID, "question": question}

    retrieval = chatbot.index_manager.retrieval_service
    original_retrieve = retrieval.retrieve

    def capture_retrieve(query: str, *args: Any, **kwargs: Any) -> Any:
        before = time.perf_counter()
        result = original_retrieve(query, *args, **kwargs)
        evidence.setdefault("retrievals", []).append(
            {
                "query": query,
                "latency_seconds": time.perf_counter() - before,
                "candidate_documents": [document_payload(d) for d in result.candidates],
                "retrieved_documents": [document_payload(d) for d in result.documents],
                "confidence": result.confidence.as_debug_payload() if result.confidence else None,
                "coverage": asdict(result.coverage) if result.coverage else None,
            }
        )
        return result

    retrieval.retrieve = capture_retrieve
    original_select = chatbot._select_and_assess_final_context

    def capture_select(query: str, documents: list[Any], policy: Any) -> Any:
        selected, coverage, assessment = original_select(query, documents, policy)
        evidence.setdefault("context_selections", []).append(
            {
                "input_documents": [document_payload(d) for d in documents],
                "selected_documents": [document_payload(d) for d in selected],
                "coverage": asdict(coverage),
                "confidence": assessment.as_debug_payload() if assessment else None,
            }
        )
        return selected, coverage, assessment

    chatbot._select_and_assess_final_context = capture_select
    original_execute = chatbot.tool_registry.execute

    def capture_execute(name: str, arguments: dict[str, Any], *, request_context: Any = None) -> Any:
        before = time.perf_counter()
        content, documents, payload = original_execute(name, arguments, request_context=request_context)
        evidence.setdefault("executions", []).append(
            {
                "name": name,
                "arguments": arguments,
                "status": payload.get("status"),
                "result_count": len(documents),
                "latency_seconds": time.perf_counter() - before,
                "documents": [document_payload(d) for d in documents],
            }
        )
        return content, documents, payload

    chatbot.tool_registry.execute = capture_execute
    original_achat = chatbot.achat

    async def capture_achat(*args: Any, **kwargs: Any) -> Any:
        result = await original_achat(*args, **kwargs)
        evidence["pipeline_result"] = {key: value for key, value in result.items() if key != "source_documents"}
        return result

    chatbot.achat = capture_achat
    started = time.perf_counter()
    try:
        response = await engine.chat(question)
        evidence["response"] = {
            "answer": response.result,
            "refused": bool(evidence.get("pipeline_result", {}).get("refused")),
            "citations": [citation.model_dump() for citation in response.citations],
            "sources": [source.model_dump() for source in response.sources],
            "tool_calls": [call.model_dump() for call in response.tool_calls],
            "tool_calling_used": response.tool_calling_used,
        }
        evidence["latency_seconds"] = time.perf_counter() - started
    finally:
        await engine.shutdown()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "refused": evidence["response"]["refused"],
                "latency_seconds": evidence["latency_seconds"],
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
