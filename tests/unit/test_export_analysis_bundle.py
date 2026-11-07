import json

import pytest

from evaluation.export_analysis_bundle import _blind_inputs, _load_runs


def test_blind_mapping_is_deterministic_and_hides_configuration() -> None:
    responses = [
        {
            "question_id": "datapizza-easy-000",
            "configuration": "dense",
            "mode": "end-to-end",
            "split": "dev",
            "generated_answer": "answer",
            "context_available": False,
            "selected_context": None,
            "citations": [],
        },
        {
            "question_id": "datapizza-easy-001",
            "configuration": "standalone",
            "mode": "end-to-end",
            "split": "dev",
            "generated_answer": "answer",
            "context_available": False,
            "selected_context": None,
            "citations": [],
        },
    ]
    questions = {
        row["question_id"]: row
        for row in (
            {
                "question_id": "datapizza-easy-000",
                "difficulty": "easy",
                "question": "q",
                "reference_answer": "r",
                "gold_passages": [],
            },
            {
                "question_id": "datapizza-easy-001",
                "difficulty": "easy",
                "question": "q",
                "reference_answer": "r",
                "gold_passages": [],
            },
        )
    }

    first, first_mapping = _blind_inputs(responses, questions, 42)
    second, second_mapping = _blind_inputs(responses, questions, 42)

    assert first == second
    assert first_mapping == second_mapping
    assert all("configuration" not in row for row in first)


def test_selected_run_must_match_its_declared_key(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"configuration": "dense", "experiment_mode": "retrieval-only", "split": "dev"}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    records = [{"question_id": f"q-{index}", "status": "success"} for index in range(12)]
    (run / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        _load_runs(tmp_path, {"hybrid/retrieval-only/dev": "run"})
