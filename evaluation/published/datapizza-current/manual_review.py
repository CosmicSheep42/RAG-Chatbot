"""Export and aggregate human answer reviews without an LLM judge."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SCORES = {"correctness", "completeness", "evidence_support"}
VALID_VALUES = {0, 1, 2, None}


def export_review(result_path: Path, output_path: Path) -> int:
    text = result_path.read_text(encoding="utf-8")
    if result_path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        configuration = None
    else:
        result = json.loads(text)
        records = result.get("records", [])
        configuration = result.get("configuration")
    rows = []
    for record in records:
        rows.append(
            {
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "difficulty": record.get("difficulty"),
                "configuration": record.get("configuration", configuration),
                "generated_answer": record.get("answer"),
                "reference_answer": record.get("manual_review", {}).get("reference_answer"),
                "gold_passages": record.get("gold_passages"),
                "retrieved_context": record.get("selected_context"),
                "citations": record.get("citations"),
                "latency": record.get("timing"),
                "review": {"correctness": None, "completeness": None, "evidence_support": None, "notes": ""},
                "review_status": "unreviewed",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def aggregate_reviews(path: Path) -> dict[str, object]:
    reviewed: list[dict[str, object]] = []
    unreviewed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        review = row.get("review", {})
        if row.get("review_status") != "reviewed":
            unreviewed += 1
            continue
        _validate_review(review)
        reviewed.append(row)

    def mean(name: str, rows: list[dict[str, object]]) -> float | None:
        values = [row["review"][name] for row in rows if row["review"].get(name) is not None]
        return statistics.mean(values) if values else None

    by_difficulty: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_configuration: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in reviewed:
        by_difficulty[str(row.get("difficulty") or "unknown")].append(row)
        by_configuration[str(row.get("configuration") or "unknown")].append(row)

    def group(rows_by_key: dict[str, list[dict[str, object]]]) -> dict[str, object]:
        return {
            key: {
                "reviewed_count": len(rows),
                "mean_correctness": mean("correctness", rows),
                "mean_completeness": mean("completeness", rows),
                "mean_evidence_support": mean("evidence_support", rows),
            }
            for key, rows in sorted(rows_by_key.items())
        }

    return {
        "reviewed_count": len(reviewed),
        "unreviewed_count": unreviewed,
        "mean_correctness": mean("correctness", reviewed),
        "mean_completeness": mean("completeness", reviewed),
        "mean_evidence_support": mean("evidence_support", reviewed),
        "by_difficulty": group(by_difficulty),
        "by_configuration": group(by_configuration),
    }


def _validate_review(review: object) -> None:
    if not isinstance(review, dict):
        raise ValueError("review must be an object")
    for name in SCORES:
        value = review.get(name)
        if value not in VALID_VALUES:
            raise ValueError(f"{name} must be 0, 1, 2, or null")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or aggregate manual benchmark answer reviews.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("result", type=Path)
    export.add_argument("output", type=Path)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("reviews", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        print(export_review(args.result, args.output))
    else:
        print(json.dumps(aggregate_reviews(args.reviews), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
