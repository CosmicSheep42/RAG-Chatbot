from local_rag_chatbot.rag.hybrid_retriever import HybridRetriever, RetrievalTuning, SimpleBM25Index, normalize_score
from local_rag_chatbot.vectorstores.contracts import ScoredDocument


def test_bm25_returns_matching_documents_in_score_order(document):
    index = SimpleBM25Index(
        [
            document("billing due date", source="billing"),
            document("unrelated onboarding text", source="onboarding"),
            document("billing invoice policy", source="invoice"),
        ]
    )

    results = index.search("billing", limit=2)

    assert [item[0].metadata["source"] for item in results] == ["billing", "invoice"]
    assert results[0][1] > 0


def test_bm25_empty_query_and_normalization_edges(document):
    index = SimpleBM25Index([document("some content")])

    assert index.search("", limit=10) == []
    assert normalize_score(1.0, 1.0, 1.0) == 1.0
    assert normalize_score(float("nan"), 0.0, 1.0) == 0.0


def test_hybrid_retrieval_deduplicates_dense_and_sparse_hits(document):
    shared = document("billing policy", source="guide", chunk_index=1)
    other = document("billing examples", source="guide", chunk_index=2)

    class Backend:
        def search(self, *_args, **_kwargs):
            return [ScoredDocument(shared, 0.9), ScoredDocument(other, 0.4)]

    retriever = HybridRetriever(
        vector_backend=Backend(),
        namespace="test",
        documents=[shared, other],
        tuning=RetrievalTuning(candidate_k=10, final_k=10),
        reranking_mode="false",
    )

    results = retriever.retrieve("billing")

    assert len(results) == 2
    assert {item.metadata["chunk_index"] for item in results} == {1, 2}
    assert all(item.metadata["retrieval_rank"] for item in results)
