# Experimental Results

## Partial Datapizza measurement

The BM25 retrieval-only configuration was executed locally with the Datapizza corpus at commit
`88ec2280d0ea4436c247fb11f3768eecabdccfe8`. It used chunk size `500`, overlap `100`, and `top_k=5`.
No generation model, embedding model, or reranker was involved.

| Split | Questions | Recall@5 | MRR@5 | nDCG@5 | Passage touch recall | Weighted coverage | Complete passage evidence | Retrieval p50 | Retrieval p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 12/12 | 0.5741 | 0.5694 | 0.5108 | 0.3569 | 0.2726 | 0.1667 | 28.0 ms | 30.3 ms |
| test | 44/44 | 0.6936 | 0.7984 | 0.6734 | 0.4392 | 0.3002 | 0.1163 | 31.3 ms | 44.4 ms |

The test split contains 43 questions with annotated passages and one question without gold passage
annotations; passage metrics use the 43-question denominator. Complete passage evidence uses the
configured 0.8 per-passage coverage threshold. These are retrieval measurements, not answer-quality
or groundedness scores.

## BM25 development generation

BM25 end-to-end was executed on the same 12-question development split with the cached
`llama3.2:3b` model through Ollama. The run completed 12/12 questions without execution failures.
Inference was CPU-only because Docker GPU passthrough was unavailable.

| Split | Questions | Retrieval p50 | Retrieval p95 | Generation p50 | Generation p95 | Total p50 | Total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 12/12 | 30.1 ms | 35.4 ms | 30.6 s | 40.0 s | 30.6 s | 40.0 s |

The benchmark recorded four heuristic incorrect-refusal signals among 12 answerable examples. This is
not a human-reviewed correctness score, and no automatic factual-accuracy score is reported. The
adapter did not expose the complete generation context for this run, so passage support and citation
quality are unavailable.

## Execution status

The standalone, dense, hybrid, hybrid-rerank, and full production RAG configurations were not executed.
The host driver is available, but Docker GPU passthrough failed because the NVIDIA Container Toolkit is
not configured. BM25 end-to-end used CPU-only Ollama successfully; the dense pilot then spent several
minutes embedding the corpus on CPU without producing a record and was stopped. No values are imputed
for the unavailable configurations or for unreviewed answer-quality metrics.

The BM25 Docker runs were saved locally at:

- `evaluation/results/datapizza/bm25/bm25/retrieval-only/run-20261107T050011Z-3a47caae` (dev)
- `evaluation/results/datapizza/bm25/bm25/retrieval-only/run-20261107T050722Z-0068c061` (test)
- `evaluation/results/datapizza/bm25/bm25/end-to-end/run-20261107T051530Z-1e748631` (dev)

These generated records are intentionally not committed. Re-run the commands in
[docs/datapizza-benchmark.md](../docs/datapizza-benchmark.md) to regenerate them.
