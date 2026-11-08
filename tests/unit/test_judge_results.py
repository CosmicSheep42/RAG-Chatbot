from evaluation.verify_judge_results import _aggregate


def test_judge_aggregate_preserves_ordinal_counts_and_both_two() -> None:
    rows = [
        {"split": "test", "configuration": "dense", "difficulty": "easy", "correctness": 2, "completeness": 2},
        {"split": "test", "configuration": "dense", "difficulty": "easy", "correctness": 1, "completeness": 0},
    ]

    result = _aggregate(rows)
    test_easy = next(row for row in result if row["split"] == "test" and row["difficulty"] == "easy")

    assert test_easy["n"] == 2
    assert test_easy["mean_correctness"] == 1.5
    assert test_easy["mean_completeness"] == 1.0
    assert test_easy["correctness_0"] == 0
    assert test_easy["correctness_1"] == 1
    assert test_easy["correctness_2"] == 1
    assert test_easy["both_2"] == 1
    assert test_easy["evidence_support_evaluable"] == 0
