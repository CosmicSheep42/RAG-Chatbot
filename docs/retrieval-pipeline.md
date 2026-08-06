# Retrieval Pipeline

Document-grounded requests follow this path:

1. Input preprocessing.
2. Static embedding-based routing.
3. Optional query rewriting using recent conversation memory.
4. Hybrid retrieval with dense Chroma search and lexical matching.
5. Optional cross-encoder reranking of retrieved candidates.
6. Context expansion and assembly.
7. Retrieval confidence assessment, with an optional retry using a broader policy.
8. Refusal when the final evidence is insufficient.
9. Answer generation, including optional document-search tool calls.

Generated indexes are stored under `data/` and can be rebuilt from the configured knowledge base.

The retriever annotates candidates with retrieval metadata such as `retrieval_score`, `retrieval_rank`, dense/sparse
scores, reranker scores, and exact-match flags. Context assembly marks retrieved candidates as `context_role="hit"` and
preserves the retrieved object when expansion finds the same chunk in the source collection. Neighboring chunks are
cloned from the source collection and marked as `context_role="neighbor"` or `context_role="structural"`; they may record
which hit caused expansion, but they do not receive copied retrieval scores.

Context selection is driven by retrieval priority: higher retrieval score first, then retrieval rank, source, chunk
position, and stable chunk identity for deterministic ties. `chunk_index` orders neighboring chunks within the same
document and acts as a tie-breaker; it is not treated as relevance. When neighboring chunks are expanded around a hit,
the hit remains the first chunk for that source and neighbors are added after it in local document order. Source
interleaving is applied after per-source expansion and is not followed by a global document-position sort. The selected
model context may include local document order for readability, but retrieval evidence is identified explicitly by
`context_role="hit"`. Broad semantic and summary contexts also reserve retrieved hits before filling the remaining
budget with document-start, middle, end, and follow-on chunks, so a relevant chunk later in a document is not dropped
just because the summary view includes earlier context.

Retrieval confidence is assessed after the final context has been assembled. If the first pass asks for retry, the retry
context is assembled and assessed separately, and the public answer/refusal decision uses the selected final context.
Only hits in the selected final context count as independent retrieval evidence. Neighbors, structural chunks, and other
auxiliary context do not increase confidence by themselves. If legacy context lacks `context_role`, the estimator falls
back to documents that carry real retrieval metadata. Empty hit sets still represent no evidence, and invalid numeric
scores are treated defensively as zero.
