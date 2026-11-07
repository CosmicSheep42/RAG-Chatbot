import json

import pytest

from evaluation.manual_review import aggregate_reviews, export_review


def test_manual_review_export_preserves_reference_answer(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "configuration": "bm25",
                "records": [
                    {
                        "question_id": "q1",
                        "question": "What is it?",
                        "difficulty": "easy",
                        "answer": "A",
                        "manual_review": {"reference_answer": "Reference"},
                        "selected_context": ["evidence"],
                        "citations": [],
                        "timing": {"generation_seconds": 0.1},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "reviews.jsonl"
    assert export_review(result, output) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["reference_answer"] == "Reference"
    assert row["review_status"] == "unreviewed"


def test_manual_review_aggregation_excludes_unreviewed_and_groups_difficulty(tmp_path):
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "difficulty": "easy",
                        "review_status": "reviewed",
                        "review": {"correctness": 2, "completeness": 1, "evidence_support": 2},
                    }
                ),
                json.dumps({"difficulty": "medium", "review_status": "unreviewed", "review": {}}),
            ]
        ),
        encoding="utf-8",
    )
    summary = aggregate_reviews(reviews)
    assert summary["reviewed_count"] == 1
    assert summary["unreviewed_count"] == 1
    assert summary["by_difficulty"]["easy"]["mean_correctness"] == 2


def test_manual_review_rejects_invalid_scores(tmp_path):
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {"review_status": "reviewed", "review": {"correctness": 3, "completeness": 1, "evidence_support": 1}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="correctness"):
        aggregate_reviews(reviews)
