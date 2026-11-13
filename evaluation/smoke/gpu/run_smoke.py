"""Run one fixed development question through the production engine, never the benchmark matrix."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time

from evaluation.adapters.datapizza import load_examples
from local_rag_chatbot.engine import create_default_engine
from local_rag_chatbot.settings import get_settings


ROOT = Path(__file__).resolve().parents[3]
QUESTION_ID = "datapizza-medium-014"
OUTPUT = Path(os.getenv("GPU_SMOKE_OUTPUT", Path(__file__).resolve().parent / "results.json"))


async def main() -> None:
    example = next(
        item for item in load_examples(ROOT / "evaluation/dataset/datapizza", "dev") if item.id == QUESTION_ID
    )
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
        max_concurrent_inferences=1,
        auto_index_on_startup=False,
    )
    engine = create_default_engine(settings)
    started = time.perf_counter()
    await engine.initialize()
    initialization_seconds = time.perf_counter() - started
    try:
        tool_started = time.perf_counter()
        _, retrieved, tool_payload = engine.services.chatbot.tool_registry.execute(
            "search_documents",
            {"query": example.question, "top_k": "10"},
        )
        tool_seconds = time.perf_counter() - tool_started
        query_started = time.perf_counter()
        response = await engine.chat(example.question)
        query_seconds = time.perf_counter() - query_started
        record = {
            "status": "success",
            "label": "single-question-smoke-test-not-benchmark",
            "question_id": example.id,
            "split": example.split,
            "question": example.question,
            "answer": response.result,
            "refused": response.result.startswith("I do not have enough reliable context"),
            "tool_calling_used": response.tool_calling_used,
            "tool_calls": [item.model_dump() for item in response.tool_calls],
            "tool_smoke": {
                "requested_top_k": "10",
                "status": tool_payload["status"],
                "result_count": len(retrieved),
                "latency_seconds": round(tool_seconds, 3),
            },
            "citations": [item.model_dump() for item in response.citations],
            "sources": [item.model_dump() for item in response.sources],
            "citation_sources_present": all(
                citation.source_name in {source.source for source in response.sources}
                for citation in response.citations
            ),
            "generation_model": settings.generation_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "initialization_seconds": round(initialization_seconds, 3),
            "query_seconds": round(query_seconds, 3),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False))
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
