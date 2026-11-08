"""Validate and import the completed Datapizza LLM-as-a-Judge evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_BUNDLE = "6c1feb34f4eae8462f628a7ec848923db5ada2b2658dc8a6835500752531695b"
CONFIGURATIONS = ("standalone", "bm25", "dense", "hybrid", "hybrid-rerank", "full")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(data: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {name}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL entry is not an object: {name}:{line_number}")
        rows.append(row)
    return rows


def _load_source(published: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mapping = json.loads((published / "blind_mapping.json").read_text(encoding="utf-8"))["mapping"]
    questions = {
        row["question_id"]: row for row in _jsonl((published / "questions.jsonl").read_bytes(), "questions.jsonl")
    }
    responses = {}
    for row in _jsonl((published / "responses.jsonl").read_bytes(), "responses.jsonl"):
        key = (row["question_id"], row["configuration"], row["mode"], row["split"])
        if key in responses:
            raise ValueError(f"duplicate published response: {key}")
        responses[key] = row
    return mapping, questions, responses


def _read_archive(
    archive: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("judge ZIP CRC validation failed")
        names = set(handle.namelist())
        required = {
            "manifest_juiz.json",
            "revisoes_llm_judge.jsonl",
            "resumo_por_cenario.json",
            "resumo_por_cenario.csv",
            "comparacoes_pareadas_test.csv",
        }
        missing = required - names
        if missing:
            raise ValueError(f"judge ZIP is missing: {sorted(missing)}")
        manifest = json.loads(handle.read("manifest_juiz.json"))
        summary = json.loads(handle.read("resumo_por_cenario.json"))
        reviews = _jsonl(handle.read("revisoes_llm_judge.jsonl"), "revisoes_llm_judge.jsonl")
        csv_rows = list(csv.DictReader(io.StringIO(handle.read("resumo_por_cenario.csv").decode("utf-8"))))
        paired = list(csv.DictReader(io.StringIO(handle.read("comparacoes_pareadas_test.csv").decode("utf-8"))))
    if manifest.get("source_zip_sha256") != EXPECTED_SOURCE_BUNDLE:
        raise ValueError("judge source bundle hash does not match the published experiment bundle")
    if manifest.get("responses_count") != 336 or len(reviews) != 336:
        raise ValueError("expected exactly 336 reviewed responses")
    return manifest, summary, reviews, {"summary": csv_rows, "paired": paired}


def _validate_reviews(
    reviews: list[dict[str, Any]],
    mapping: dict[str, str],
    questions: dict[str, dict[str, Any]],
    responses: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    if len({row.get("blind_response_id") for row in reviews}) != 336:
        raise ValueError("blind response IDs are not unique")
    counts: Counter[tuple[str, str]] = Counter()
    for row in reviews:
        blind_id = row.get("blind_response_id")
        if blind_id not in mapping:
            raise ValueError(f"unknown blind response ID: {blind_id}")
        configuration, mode, split, question_id = mapping[blind_id].split("/")
        if configuration not in CONFIGURATIONS or mode != "end-to-end":
            raise ValueError(f"invalid mapping for {blind_id}")
        if (row.get("configuration"), row.get("split"), row.get("question_id")) != (configuration, split, question_id):
            raise ValueError(f"review does not match blind mapping: {blind_id}")
        if question_id not in questions:
            raise ValueError(f"unknown question ID: {question_id}")
        if row.get("review_status") != "reviewed" or row.get("reviewer_type") != "llm-as-a-judge":
            raise ValueError(f"review metadata is invalid: {blind_id}")
        for field in ("correctness", "completeness"):
            if row.get(field) not in (0, 1, 2):
                raise ValueError(f"{field} must be 0, 1, or 2: {blind_id}")
        if row.get("evidence_support") is not None:
            raise ValueError(f"evidence support must remain null: {blind_id}")
        key = (question_id, configuration, mode, split)
        source = responses.get(key)
        if source is None or row.get("generated_answer") != source.get("generated_answer"):
            raise ValueError(f"generated answer mismatch: {blind_id}")
        counts[(configuration, split)] += 1
    for configuration in CONFIGURATIONS:
        if counts[(configuration, "dev")] != 12 or counts[(configuration, "test")] != 44:
            raise ValueError(f"unexpected split counts for {configuration}")


def _aggregate(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        groups[(row["split"], row["configuration"], row["difficulty"])].append(row)
    groups.update({})
    for split in ("dev", "test", "all"):
        for configuration in CONFIGURATIONS:
            rows = [
                row
                for row in reviews
                if row["configuration"] == configuration and (split == "all" or row["split"] == split)
            ]
            groups[(split, configuration, "all")] = rows
            for difficulty in ("easy", "medium"):
                groups[(split, configuration, difficulty)] = [row for row in rows if row["difficulty"] == difficulty]
    output = []
    for (split, configuration, difficulty), rows in sorted(groups.items()):
        if not rows:
            continue
        correctness = [row["correctness"] for row in rows]
        completeness = [row["completeness"] for row in rows]
        output.append(
            {
                "split": split,
                "configuration": configuration,
                "difficulty": difficulty,
                "n": len(rows),
                "mean_correctness": round(statistics.mean(correctness), 6),
                "mean_completeness": round(statistics.mean(completeness), 6),
                "correctness_0": correctness.count(0),
                "correctness_1": correctness.count(1),
                "correctness_2": correctness.count(2),
                "completeness_0": completeness.count(0),
                "completeness_1": completeness.count(1),
                "completeness_2": completeness.count(2),
                "both_2": sum(row["correctness"] == 2 and row["completeness"] == 2 for row in rows),
                "evidence_support_evaluable": 0,
            }
        )
    return output


def validate_archive(archive: Path, published: Path) -> dict[str, Any]:
    manifest, declared_summary, reviews, csv_data = _read_archive(archive)
    mapping, questions, responses = _load_source(published)
    _validate_reviews(reviews, mapping, questions, responses)
    calculated = _aggregate(reviews)
    declared_by_key = {(row["split"], row["configuration"], row["difficulty"]): row for row in declared_summary}
    for row in calculated:
        declared = declared_by_key.get((row["split"], row["configuration"], row["difficulty"]))
        if declared is None:
            raise ValueError(f"missing declared aggregate: {row}")
        for field in ("n", "correctness_0", "correctness_1", "correctness_2", "both_2"):
            if int(declared[field]) != row[field]:
                raise ValueError(
                    f"aggregate mismatch for {field}: {row['split']}/{row['configuration']}/{row['difficulty']}"
                )
        for field in ("mean_correctness", "mean_completeness"):
            if not math.isclose(float(declared[field]), row[field], abs_tol=1e-6):
                raise ValueError(
                    f"aggregate mismatch for {field}: {row['split']}/{row['configuration']}/{row['difficulty']}"
                )
    return {"manifest": manifest, "reviews": reviews, "aggregates": calculated, "paired": csv_data["paired"]}


def import_results(validation: dict[str, Any], output: Path, archive: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    reviews = validation["reviews"]
    compact = []
    for row in reviews:
        compact.append(
            {
                key: row.get(key)
                for key in (
                    "blind_response_id",
                    "question_id",
                    "split",
                    "difficulty",
                    "configuration",
                    "correctness",
                    "completeness",
                    "evidence_support",
                    "reasoning_summary",
                    "reviewer_type",
                    "review_status",
                )
            }
        )
    (output / "reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in compact), encoding="utf-8"
    )
    aggregates = validation["aggregates"]
    test_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        if row["split"] == "test":
            test_by_question[row["question_id"]].append(row)
    all_zero = sum(
        len(rows) == len(CONFIGURATIONS) and all(row["correctness"] == 0 for row in rows)
        for rows in test_by_question.values()
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "reviewer": "OpenAI GPT-6 via ChatGPT",
                "evaluation_type": "LLM-as-a-Judge",
                "review_mode": "single-pass blinded evaluation",
                "human_adjudication": False,
                "evidence_support_evaluable": 0,
                "evidence_support_unavailable_count": len(reviews),
                "test_questions_all_six_correctness_zero": all_zero,
                "aggregates": aggregates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fields = list(aggregates[0])
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)
    with (output / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        paired = validation["paired"]
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]) if paired else ["split"])
        writer.writeheader()
        writer.writerows(paired)
    source_hash = _sha256(archive)
    manifest = {
        **validation["manifest"],
        "judge_archive_sha256": source_hash,
        "imported_review_count": len(compact),
        "published_source": "evaluation/published/datapizza/",
        "human_adjudication": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test = [row for row in aggregates if row["split"] == "test" and row["difficulty"] == "all"]
    lines = [
        "# Datapizza LLM-as-a-Judge evaluation",
        "",
        "This is a single-pass blinded evaluation by OpenAI GPT-6 via ChatGPT, not a human review or expert validation.",
        "Evidence support was not evaluated and remains unavailable for all 336 responses.",
        "",
        "## Held-out test split",
        "",
        "| Configuration | Correctness mean / 2 | Completeness mean / 2 | Correctness 2/2 | Both 2/2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in test:
        lines.append(
            f"| {row['configuration']} | {row['mean_correctness']:.3f} | {row['mean_completeness']:.3f} | {row['correctness_2']}/{row['n']} | {row['both_2']}/{row['n']} |"
        )
    lines += [
        "",
        "Scores use an ordinal 0–2 rubric and must not be read as accuracy percentages. The results are limited by one judge, no human adjudication, one D&D 5e domain, 56 questions, and potentially ambiguous reference criteria for medium questions 001, 011, 013, 019, and 028. Medium question 004 has uncertain evidence annotations and lacks gold passages.",
        f"On the held-out test split, {all_zero} questions received correctness 0 from all six systems. Evidence support was unavailable for all {len(reviews)} reviewed responses.",
        "",
        "The highest average correctness in this evaluation was observed for Hybrid + Rerank. This supports a finding on this fixed benchmark only; it does not establish universal superiority or isolate reranking as the sole cause. Groundedness and claim-level citation support remain unmeasured.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Datapizza judge results\n\nScores are from a single-pass blinded LLM-as-a-Judge evaluation by OpenAI GPT-6 via ChatGPT. They are not human-reviewed. `reviews.jsonl` contains compact normalized records linked to the original blind IDs and published benchmark records. Evidence support is unavailable for every response. See `report.md` and `manifest.json`.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and import Datapizza LLM-as-a-Judge results.")
    parser.add_argument("--archive", type=Path, default=Path("datapizza_llm_judge_resultados.zip"))
    parser.add_argument("--published", type=Path, default=Path("evaluation/published/datapizza"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/judge/datapizza"))
    parser.add_argument("--import-results", action="store_true")
    args = parser.parse_args()
    validation = validate_archive(args.archive, args.published)
    if args.import_results:
        import_results(validation, args.output, args.archive)
    print(
        json.dumps(
            {
                "status": "ok",
                "reviews": len(validation["reviews"]),
                "aggregates": len(validation["aggregates"]),
                "imported": args.import_results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
