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
