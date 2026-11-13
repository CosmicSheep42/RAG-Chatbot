from types import SimpleNamespace

from local_rag_chatbot.rag.retrieval_confidence import RetrievalConfidenceEstimator, RetrievalDecision
from local_rag_chatbot.rag.retrieval_policy import QueryIntent, RetrievalPolicy


def policy(intent=QueryIntent.FACTUAL_LOOKUP):
    return RetrievalPolicy(intent, 5, 5, False, 1, 5, 1, 0.2, 10, 10)


def test_empty_evidence_refuses_without_inventing_confidence():
    assessment = RetrievalConfidenceEstimator().assess([], SimpleNamespace(sufficient=False, score=0), policy())

    assert assessment.decision is RetrievalDecision.REFUSE
    assert assessment.confidence == 0
    assert "NO_EVIDENCE" in assessment.reason_codes


def test_strong_multi_signal_evidence_is_answerable(document):
    candidate = document(
        "The answer",
        context_role="hit",
        retrieval_sparse_score=4,
        retrieval_dense_score=0.8,
        retrieval_reranker_score=3,
        retrieval_dense_rank=1,
        retrieval_sparse_rank=1,
    )

    assessment = RetrievalConfidenceEstimator().assess(
        [candidate], SimpleNamespace(sufficient=True, score=1.0), policy()
    )

    assert assessment.decision is RetrievalDecision.ANSWER
    assert assessment.confidence >= 0.62


def test_confidence_uses_best_supported_evidence_independent_of_context_order(document):
    sparse_only = document("Sparse only", context_role="hit", retrieval_sparse_score=20, retrieval_sparse_rank=3)
    candidate = document(
        "Retrieved evidence",
        context_role="hit",
        retrieval_sparse_score=20,
        retrieval_dense_score=0.8,
        retrieval_dense_rank=1,
        retrieval_sparse_rank=1,
    )

    assessment = RetrievalConfidenceEstimator().assess(
        [sparse_only, candidate],
        SimpleNamespace(sufficient=True, score=0.5),
        policy(QueryIntent.COMPARISON),
    )

    assert assessment.decision is RetrievalDecision.ANSWER
    assert "INSUFFICIENT_COVERAGE" not in assessment.reason_codes

    insufficient = RetrievalConfidenceEstimator().assess(
        [sparse_only, candidate],
        SimpleNamespace(sufficient=False, score=1.0),
        policy(QueryIntent.COMPARISON),
    )

    assert insufficient.decision is RetrievalDecision.RETRY
    assert "INSUFFICIENT_COVERAGE" in insufficient.reason_codes


def test_dense_rank_alone_is_not_semantic_relevance(document):
    candidate = document("unrelated", context_role="hit", retrieval_dense_rank=1)

    assessment = RetrievalConfidenceEstimator().assess(
        [candidate], SimpleNamespace(sufficient=True, score=1.0), policy()
    )

    assert assessment.confidence == 0.0
    assert assessment.decision is RetrievalDecision.REFUSE


def test_configured_confidence_threshold_affects_answer_decision(document):
    candidate = document(
        "The answer",
        context_role="hit",
        retrieval_dense_score=0.8,
        retrieval_sparse_score=4,
    )
    estimator = RetrievalConfidenceEstimator()
    estimator.set_minimum_confidence(0.99)

    assessment = estimator.assess([candidate], SimpleNamespace(sufficient=True, score=1.0), policy())

    assert assessment.decision is RetrievalDecision.RETRY


def test_strong_dense_only_evidence_can_answer_without_reranker(document):
    candidate = document(
        "The application restricts concurrent connections to eight.",
        context_role="hit",
        retrieval_dense_score=0.95,
    )

    assessment = RetrievalConfidenceEstimator().assess(
        [candidate], SimpleNamespace(sufficient=True, score=1.0), policy()
    )

    assert assessment.decision is RetrievalDecision.ANSWER
    assert "STRONG_DENSE_EVIDENCE" in assessment.reason_codes
