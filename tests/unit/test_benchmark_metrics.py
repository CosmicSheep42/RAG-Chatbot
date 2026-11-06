import math

from evaluation.metrics import (
    abstention_result,
    citation_quality,
    complete_evidence_retrieval,
    fact_coverage,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_retrieval_metrics_deduplicate_chunks_and_score_multiple_documents():
    results = [
        {"document_id": "guide.md", "rank": 1},
        {"document_id": "guide.md", "rank": 2},
        {"document_id": "operations.md", "rank": 3},
    ]

    assert recall_at_k(results, ["guide.md", "operations.md"], 3) == 1.0
    assert reciprocal_rank_at_k(results, ["operations.md"], 3) == 1 / 2
    assert complete_evidence_retrieval(results, ["guide.md", "operations.md"], 3)
    assert ndcg_at_k(results, ["guide.md", "operations.md"], 3) == 1.0


def test_metrics_handle_empty_and_unanswerable_inputs():
    assert recall_at_k([], [], 5) != recall_at_k([], [], 5)
    assert math.isnan(recall_at_k([], [], 5))
    assert reciprocal_rank_at_k([], ["guide.md"], 5) == 0.0
    assert not complete_evidence_retrieval([], ["guide.md"], 5)
    assert abstention_result("I do not have enough reliable context.", False) == "correct_abstention"
    assert abstention_result("The answer is 42.", False) == "unsupported_answer"
    assert abstention_result("I cannot answer.", True) == "incorrect_refusal"


def test_answer_and_citation_metrics_are_separate():
    assert fact_coverage("The limit is 50 active API keys.", ["The limit is 50 active API keys."]) == 1.0
    assert citation_quality(["guide.md", "missing.md"], ["guide.md"], ["guide.md"]) == {
        "validity": 0.5,
        "relevance": 0.5,
    }
