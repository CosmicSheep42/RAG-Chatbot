"""Export selected Datapizza benchmark runs for audit and answer review."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_SELECTION = {
    "bm25/retrieval-only/dev": "bm25/bm25/retrieval-only/run-20261107T050701Z-2ae00fcc",
    "bm25/retrieval-only/test": "bm25/bm25/retrieval-only/run-20261107T050722Z-0068c061",
    "dense/retrieval-only/dev": "dense/dense/retrieval-only/run-gpu-dev-dense",
    "dense/retrieval-only/test": "dense/dense/retrieval-only/run-gpu-test-dense",
    "hybrid/retrieval-only/dev": "hybrid/hybrid/retrieval-only/run-gpu-dev-hybrid",
    "hybrid/retrieval-only/test": "hybrid/hybrid/retrieval-only/run-gpu-test-hybrid",
    "hybrid-rerank/retrieval-only/dev": "hybrid-rerank/hybrid-rerank/retrieval-only/run-gpu-dev-hybrid-rerank",
    "hybrid-rerank/retrieval-only/test": "hybrid-rerank/hybrid-rerank/retrieval-only/run-gpu-test-hybrid-rerank",
    "standalone/end-to-end/dev": "standalone/standalone/end-to-end/run-gpu-dev-standalone",
    "standalone/end-to-end/test": "standalone/standalone/end-to-end/run-gpu-test-standalone",
    "bm25/end-to-end/dev": "bm25/bm25/end-to-end/run-gpu-dev-bm25-e2e",
    "bm25/end-to-end/test": "bm25/bm25/end-to-end/run-gpu-test-bm25-e2e",
    "dense/end-to-end/dev": "dense/dense/end-to-end/run-gpu-dev-dense-e2e",
    "dense/end-to-end/test": "dense/dense/end-to-end/run-gpu-test-dense-e2e",
    "hybrid/end-to-end/dev": "hybrid/hybrid/end-to-end/run-gpu-dev-hybrid-e2e",
    "hybrid/end-to-end/test": "hybrid/hybrid/end-to-end/run-gpu-test-hybrid-e2e",
    "hybrid-rerank/end-to-end/dev": "hybrid-rerank/hybrid-rerank/end-to-end/run-gpu-dev-hybrid-rerank-e2e",
    "hybrid-rerank/end-to-end/test": "hybrid-rerank/hybrid-rerank/end-to-end/run-gpu-test-hybrid-rerank-e2e",
    "full/end-to-end/dev": "full/full/end-to-end/run-gpu-dev-full",
    "full/end-to-end/test": "full/full/end-to-end/run-gpu-test-full",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _load_selection(path: Path | None) -> dict[str, str]:
    if path is None:
        return DEFAULT_SELECTION.copy()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("selection must be a non-empty object")
    return {str(key): str(value) for key, value in data.items()}


def _load_runs(results: Path, selection: dict[str, str]) -> list[dict[str, Any]]:
    runs = []
    for key, relative in sorted(selection.items()):
        parts = key.split("/")
        if len(parts) != 3:
            raise ValueError(f"invalid selection key: {key}")
        configuration, mode, split = parts
        run_dir = results / relative
        manifest_path = run_dir / "manifest.json"
        records_path = run_dir / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            raise FileNotFoundError(f"incomplete selected run: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("configuration"), manifest.get("experiment_mode"), manifest.get("split")) != (
            configuration,
            mode,
            split,
        ):
            raise ValueError(f"manifest does not match selection key: {key}")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = 12 if split == "dev" else 44
        ids = [record.get("question_id") for record in records]
        if (
            len(records) != expected
            or len(set(ids)) != len(ids)
            or any(record.get("status") != "success" for record in records)
        ):
            raise ValueError(f"selected run is incomplete or invalid: {run_dir}")
        runs.append(
            {
                "configuration": configuration,
                "mode": mode,
                "split": split,
                "path": run_dir,
                "manifest": manifest,
                "summary": summary,
                "records": records,
            }
        )
    return runs


def _questions(dataset: Path, split_manifest: Path) -> list[dict[str, Any]]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_by_id = {question_id: split for split in ("dev", "test") for question_id in manifest[split]}
    rows = []
    for tier in ("easy", "medium"):
        for item in json.loads((dataset / f"{tier}.json").read_text(encoding="utf-8")):
            question_id = f"datapizza-{tier}-{int(item['id']):03d}"
            rows.append(
                {
                    "question_id": question_id,
                    "split": split_by_id[question_id],
                    "difficulty": tier,
                    "question": item["question"],
                    "reference_answer": item["answer"],
                    "gold_passages": item["passages"],
                }
            )
    return sorted(rows, key=lambda row: row["question_id"])


def _responses(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run["mode"] != "end-to-end":
            continue
        for record in run["records"]:
            rows.append(
                {
                    "question_id": record["question_id"],
                    "configuration": run["configuration"],
                    "mode": run["mode"],
                    "split": run["split"],
                    "difficulty": record.get("difficulty"),
                    "generated_answer": record.get("answer"),
                    "status": record.get("status"),
                    "retrieved_documents": record.get("retrieved_documents"),
                    "selected_context": record.get("selected_context"),
                    "context_available": record.get("context_available"),
                    "citations": record.get("citations"),
                    "route": record.get("route"),
                    "refused": record.get("refused"),
                    "citation_identifiers": record.get(
                        "citation_identifiers", re.findall(r"\[(S\d+)\]", record.get("answer") or "")
                    ),
                    "available_citation_identifiers": record.get("available_citation_identifiers", []),
                    "invalid_citation_identifiers": record.get("invalid_citation_identifiers", []),
                    "pipeline_trace": record.get("pipeline_trace"),
                    "retrieval_confidence_assessment": record.get("retrieval_confidence_assessment"),
                    "partial_answer": record.get("partial_answer"),
                    "tool_calls": record.get("tool_calls", []),
                    "recovery_attempts": record.get("recovery_attempts"),
                    "query_rewrite": record.get("query_rewrite"),
                    "timing": record.get("timing", {}),
                    "metrics": record.get("metrics", {}),
                }
            )
    return sorted(rows, key=lambda row: (row["split"], row["question_id"], row["configuration"], row["mode"]))


def _evidence(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for record in run["records"]:
            rows.append(
                {
                    "question_id": record["question_id"],
                    "configuration": run["configuration"],
                    "mode": run["mode"],
                    "split": run["split"],
                    "gold_passages": record.get("gold_passages"),
                    "retrieved_documents": record.get("retrieved_documents"),
                    "selected_context": record.get("selected_context"),
                    "context_available": record.get("context_available"),
                    "citations": record.get("citations"),
                    "citation_identifiers": record.get(
                        "citation_identifiers", re.findall(r"\[(S\d+)\]", record.get("answer") or "")
                    ),
                    "available_citation_identifiers": record.get("available_citation_identifiers", []),
                    "invalid_citation_identifiers": record.get("invalid_citation_identifiers", []),
                    "pipeline_trace": record.get("pipeline_trace"),
                    "retrieval_confidence_assessment": record.get("retrieval_confidence_assessment"),
                    "partial_answer": record.get("partial_answer"),
                    "tool_calls": record.get("tool_calls", []),
                    "recovery_attempts": record.get("recovery_attempts"),
                    "query_rewrite": record.get("query_rewrite"),
                }
            )
    return sorted(rows, key=lambda row: (row["split"], row["question_id"], row["configuration"], row["mode"]))


def _blind_inputs(
    responses: list[dict[str, Any]], questions: dict[str, dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates = [row for row in responses if row["mode"] == "end-to-end"]
    randomizer = random.Random(seed)
    shuffled = list(candidates)
    randomizer.shuffle(shuffled)
    mapping = {}
    inputs = []
    for index, row in enumerate(shuffled, 1):
        blind_id = f"response-{index:06d}"
        mapping[blind_id] = f"{row['configuration']}/{row['mode']}/{row['split']}/{row['question_id']}"
        question = questions[row["question_id"]]
        inputs.append(
            {
                "blind_response_id": blind_id,
                "question_id": row["question_id"],
                "split": row["split"],
                "difficulty": question["difficulty"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "generated_answer": row["generated_answer"],
                "gold_passages": question["gold_passages"],
                "retrieved_evidence": row["selected_context"] if row["context_available"] else None,
                "citations": row["citations"],
            }
        )
    return inputs, mapping


def _review_template(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "blind_response_id": row["blind_response_id"],
            "question_id": row["question_id"],
            "correctness": None,
            "completeness": None,
            "evidence_support": None,
            "reasoning_summary": "",
            "evidence_references": [],
            "review_status": "unreviewed",
            "reviewer_type": None,
        }
        for row in inputs
    ]


def export_bundle(
    results: Path,
    dataset: Path,
    corpus: Path,
    published: Path,
    zip_path: Path | None,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    selection = _load_selection(selection_path)
    runs = _load_runs(results, selection)
    questions = _questions(dataset, dataset / "splits.json")
    question_map = {row["question_id"]: row for row in questions}
    responses = _responses(runs)
    with tempfile.TemporaryDirectory(prefix="datapizza-publish-") as temporary:
        staging = Path(temporary)
        if published.exists():
            shutil.rmtree(published)
        published.mkdir(parents=True, exist_ok=True)
        _write_jsonl(published / "questions.jsonl", questions)
        _write_jsonl(published / "responses.jsonl", responses)
        _write_jsonl(published / "evidence.jsonl", _evidence(runs))
        blind, mapping = _blind_inputs(responses, question_map, 42)
        _write_jsonl(published / "judge_input.jsonl", blind)
        _write_jsonl(published / "review_template.jsonl", _review_template(blind))
        _write_json(published / "blind_mapping.json", {"seed": 42, "mapping": mapping})
        shutil.copy2("evaluation/manual_review.py", published / "manual_review.py")
        _write_json(
            published / "experiments.json",
            [
                {
                    "configuration": run["configuration"],
                    "mode": run["mode"],
                    "split": run["split"],
                    "source_run": str(run["path"]),
                    "manifest": run["manifest"],
                }
                for run in runs
            ],
        )
        _write_json(published / "selection.json", selection)
        for name in ("environment.json", "blocked-scenarios.json"):
            source = results / name
            if source.is_file():
                shutil.copy2(source, published / name)
        for run in runs:
            name = f"{run['configuration']}-{run['mode']}-{run['split']}.jsonl"
            _write_jsonl(
                published / ("generation" if run["mode"] == "end-to-end" else "retrieval") / name, run["records"]
            )
        _write_json(
            published / "summary.json",
            {
                "question_count": len(questions),
                "response_count": len(responses),
                "retrieval_record_count": sum(len(run["records"]) for run in runs if run["mode"] == "retrieval-only"),
                "runs": len(runs),
                "run_summaries": [
                    {
                        "configuration": run["configuration"],
                        "mode": run["mode"],
                        "split": run["split"],
                        "record_file": f"{run['configuration']}-{run['mode']}-{run['split']}.jsonl",
                        "declared_summary": run["summary"],
                    }
                    for run in runs
                ],
                "human_review": "pending",
            },
        )
        (published / "README.md").write_text(
            "# Published Datapizza benchmark records\n\nThese files are exact per-question records selected from local completed runs. Generated answers are unchanged. `judge_input.jsonl` hides configuration names; use `blind_mapping.json` only after scoring. `review_template.jsonl` is unscored. Existing records under `evaluation/judge/datapizza/` belong only to the historical publication and do not score these answers.\n",
            encoding="utf-8",
        )
        snapshot = Path("evaluation/RESULTS.md").read_text(encoding="utf-8")
        snapshot = snapshot.replace("(published/datapizza/)", "(../datapizza/)")
        snapshot = snapshot.replace("(published/datapizza-current/)", "(./)")
        snapshot = snapshot.replace(
            "(comparisons/baseline-vs-current/REPORT.md)", "(../../comparisons/baseline-vs-current/REPORT.md)"
        )
        snapshot = snapshot.replace("(judge/datapizza/)", "(../../judge/datapizza/)")
        snapshot = snapshot.replace("(../docs/datapizza-benchmark.md)", "(../../../docs/datapizza-benchmark.md)")
        (published / "RESULTS.md").write_text(snapshot, encoding="utf-8")
        (published / "checksums.sha256").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(published).as_posix()}\n"
                for path in sorted(published.rglob("*"))
                if path.is_file()
            ),
            encoding="utf-8",
        )
        if zip_path is None:
            return {
                "published": str(published),
                "questions": len(questions),
                "responses": len(responses),
                "runs": len(runs),
            }
        if zip_path.exists():
            raise FileExistsError(f"refusing to overwrite existing ZIP: {zip_path}")
        if zip_path.parent:
            zip_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_root = staging / "datapizza-rag-evaluation"
        (bundle_root / "dataset" / "corpus").mkdir(parents=True)
        for source in sorted(corpus.glob("*.md")):
            shutil.copy2(source, bundle_root / "dataset" / "corpus" / source.name)
        for name in ("easy.json", "medium.json", "splits.json"):
            shutil.copy2(dataset / name, bundle_root / "dataset" / name)
        shutil.copytree(published, bundle_root / "published")
        shutil.copy2("THIRD_PARTY_NOTICES.md", bundle_root / "THIRD_PARTY_NOTICES.md")
        (bundle_root / "reports").mkdir()
        shutil.copy2("evaluation/RESULTS.md", bundle_root / "reports" / "RESULTS.md")
        manifest = {
            "dataset": "Datapizza AI Lab D&D 5e RAG Evaluation Dataset",
            "source_revision": json.loads((dataset / "splits.json").read_text())["source_revision"],
            "question_count": len(questions),
            "response_count": len(responses),
            "run_count": len(runs),
            "note": "Generated answers are exact records; no LLM inference or scoring was performed by this exporter.",
        }
        _write_json(bundle_root / "manifest.json", manifest)
        files = sorted(path for path in bundle_root.rglob("*") if path.is_file())
        (bundle_root / "checksums.sha256").write_text(
            "".join(f"{_sha256(path)}  {path.relative_to(bundle_root).as_posix()}\n" for path in files),
            encoding="utf-8",
        )
        with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
    return {
        "published": str(published),
        "zip": str(zip_path),
        "questions": len(questions),
        "responses": len(responses),
        "runs": len(runs),
        "zip_sha256": _sha256(zip_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export selected Datapizza benchmark records without running inference."
    )
    parser.add_argument("--results", type=Path, default=Path("evaluation/results/datapizza"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/datapizza"))
    parser.add_argument("--corpus", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--published", type=Path, default=Path("evaluation/published/datapizza"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-zip", action="store_true", help="publish records without creating the local ZIP")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_bundle(
                args.results,
                args.dataset,
                args.corpus,
                args.published,
                None if args.no_zip else args.output,
                args.selection,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
