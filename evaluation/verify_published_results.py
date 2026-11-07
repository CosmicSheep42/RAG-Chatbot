"""Verify published Datapizza benchmark records and recompute aggregates offline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record.get("metrics", {}).get(key) for record in records]
    values = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(value)]
    return statistics.mean(values) if values else None


def verify(published: Path) -> dict[str, Any]:
    summary = json.loads((published / "summary.json").read_text(encoding="utf-8"))
    checked_runs = 0
    for run in summary["run_summaries"]:
        folder = "generation" if run["mode"] == "end-to-end" else "retrieval"
        path = published / folder / run["record_file"]
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        declared = run["declared_summary"]
        if len(records) != declared["successful_count"] + declared["failed_count"]:
            raise ValueError(f"record count mismatch: {path}")
        if any(record.get("status") != "success" for record in records):
            raise ValueError(f"published run contains a non-success record: {path}")
        if run["mode"] == "retrieval-only":
            for summary_key, metric_key in (
                ("recall_at_k", "recall_at_k"),
                ("mrr_at_k", "mrr_at_k"),
                ("ndcg_at_k", "ndcg_at_k"),
                ("passage_touch_recall_mean", "passage_recall_at_k"),
                ("weighted_passage_coverage_mean", "passage_coverage_at_k"),
            ):
                expected = declared.get(summary_key)
                actual = _mean(records, metric_key)
                if expected is not None and (
                    actual is None or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
                ):
                    raise ValueError(f"aggregate mismatch for {summary_key}: {path}")
        checked_runs += 1
    return {
        "status": "ok",
        "questions": summary["question_count"],
        "responses": summary["response_count"],
        "runs_checked": checked_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify published Datapizza benchmark records offline.")
    parser.add_argument("published", type=Path, nargs="?", default=Path("evaluation/published/datapizza"))
    args = parser.parse_args()
    print(json.dumps(verify(args.published), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
