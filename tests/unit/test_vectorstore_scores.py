from local_rag_chatbot.vectorstores.chroma_backend import ChromaVectorStoreBackend
from local_rag_chatbot.vectorstores.pgvector_backend import PgVectorStoreBackend


def test_chroma_mmr_preserves_similarity_score(document):
    selected = document("relevant", source="guide.md", chunk_id="1")

    class Retriever:
        def invoke(self, _query):
            return [selected]

    class Store:
        def as_retriever(self, **_kwargs):
            return Retriever()

        def similarity_search_with_relevance_scores(self, _query, k):
            assert k == 4
            return [(selected, 0.87)]

    backend = object.__new__(ChromaVectorStoreBackend)
    backend._load_store = lambda _namespace: Store()
    results = backend.search("rag", "question", k=1, fetch_k=4, strategy="mmr")

    assert results[0].score == 0.87
    assert results[0].document.metadata["retrieval_dense_score"] == 0.87


def test_pgvector_mmr_keeps_query_relevance_separate_from_rank(document):
    backend = object.__new__(PgVectorStoreBackend)
    backend._cosine_similarity = lambda left, right: sum(a * b for a, b in zip(left, right))
    first = {"document": document("first"), "score": 0.8, "embedding": [1.0, 0.0]}
    second = {"document": document("second"), "score": 0.2, "embedding": [0.0, 1.0]}

    selected = backend._mmr_candidates([first, second], [1.0, 0.0], k=2, lambda_mult=0.5)

    assert [item["score"] for item in selected] == [0.8, 0.2]
    assert selected[0]["document"].metadata["retrieval_dense_rank"] == 1
