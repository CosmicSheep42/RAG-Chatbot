# Datapizza Experimental Results

## Experimental setup

The runs below were executed locally on Debian 13 with an NVIDIA RTX 3050 Laptop GPU (4 GB), Docker GPU passthrough, Ollama `0.32.1`, `llama3.2:3b`, and `nomic-embed-text`. Llama was partially offloaded by Ollama (34% CPU / 66% GPU); embeddings used the GPU. The reranker was `BAAI/bge-reranker-base` on CPU.

The Datapizza corpus used 20 Markdown documents, chunk size `500`, overlap `100`, `top_k=5`, Chroma namespace `rag`, temperature `0`, and one inference at a time. The development split has 12 questions and the held-out test split has 44. One test question has no annotated gold passages, so passage metrics use 43 evaluable questions there. All runs used unique local run directories; raw records remain under `evaluation/results/datapizza/` and are not committed.

## Retrieval-only comparison

| Split / metric | BM25 | Dense | Hybrid | Hybrid + rerank |
|---|---:|---:|---:|---:|
| dev Recall@5 | 0.5741 | 0.5440 | 0.6227 | 0.7199 |
| dev MRR@5 | 0.5694 | 0.7222 | 0.7917 | 0.8361 |
| dev nDCG@5 | 0.5108 | 0.5650 | 0.6363 | 0.7062 |
| dev weighted passage coverage | 0.2726 | 0.2858 | 0.2963 | 0.2994 |
| dev retrieval p50 | 28.0 ms | 29.2 ms | 79.4 ms | 1.19 s |
| test Recall@5 | 0.6936 | 0.6605 | 0.7446 | 0.7576 |
| test MRR@5 | 0.7984 | 0.7907 | 0.8624 | 0.8605 |
| test nDCG@5 | 0.6734 | 0.6655 | 0.7360 | 0.7592 |
| test weighted passage coverage | 0.3002 | 0.2472 | 0.3298 | 0.4017 |
| test retrieval p50 | 31.3 ms | 35.7 ms | 89.8 ms | 1.19 s |

Complete passage evidence rates on test were BM25 `0.1163`, Dense `0.1163`, Hybrid `0.1163`, and Hybrid + rerank `0.2093`. The reranker improved measured retrieval metrics on this fixed corpus, but increased retrieval latency substantially.

## End-to-end comparison

Every listed run completed without execution failures. Values are p50/p95 seconds for the held-out test split; development runs are preserved in the corresponding run directories.

| Configuration | Questions | Retrieval p50 | Generation p50 | Total p50 | Total p95 |
|---|---:|---:|---:|---:|---:|
| Standalone | 44/44 | N/A | 4.12 | 4.12 | 12.09 |
| BM25 RAG | 44/44 | 0.038 | 7.44 | 7.48 | 16.22 |
| Dense RAG | 44/44 | 1.824 | 12.80 | 14.62 | 23.16 |
| Hybrid RAG | 44/44 | 1.737 | 11.25 | 13.11 | 21.49 |
| Hybrid + rerank RAG | 44/44 | 2.727 | 9.63 | 12.34 | 18.01 |
| Full production RAG | 44/44 | N/A | N/A | 7.88 | 13.48 |

The Full pipeline exposes total engine latency and application-level citation validity, but not separate retrieval and generation timings in these records. Its citation validity was `1.0`; this means returned source identifiers were valid, not that every answer claim was manually verified.

Exploratory correctness and completeness scores are published separately in the completed LLM-as-a-Judge import below. They are ordinal judgments from one blinded LLM reviewer, not independently verified accuracy. Human-adjudicated correctness, claim-level groundedness, and evidence-support scores are not available.

## Auditable records and answer review

Selected per-question records are published under [evaluation/published/datapizza/](published/datapizza/). They were derived from the completed local run directories without changing generated answers. The original local run directories remain outside version control. The deterministic export command is:

```bash
.venv/bin/python -m evaluation.export_analysis_bundle \
  --results evaluation/results/datapizza \
  --dataset evaluation/dataset/datapizza \
  --corpus knowledge-base \
  --published evaluation/published/datapizza \
  --output exports/datapizza-rag-evaluation-bundle.zip
```

The published data includes per-question retrieval and generation records, source manifests, checksums, and the original unscored blind-review input. The completed judge import is separate under [evaluation/judge/datapizza/](judge/datapizza/), so future bundle exports cannot overwrite it.

The LLM-as-a-Judge test-split summary is:

| Configuration | Correctness mean / 2 | Completeness mean / 2 | Correctness 2/2 |
|---|---:|---:|---:|
| Standalone | 0.091 | 0.091 | 0/44 |
| BM25 RAG | 0.614 | 0.500 | 8/44 |
| Dense RAG | 0.795 | 0.636 | 10/44 |
| Hybrid RAG | 0.705 | 0.500 | 10/44 |
| Hybrid + Rerank RAG | 0.886 | 0.705 | 11/44 |
| Full RAG | 0.727 | 0.614 | 10/44 |

These are ordinal 0–2 scores from one single-pass blinded LLM reviewer, not factual accuracy percentages. Human adjudication was not performed, evidence support was not evaluated for any of the 336 answers, and the judge may have made factual mistakes. Medium questions `001`, `011`, `013`, `019`, and `028` have potentially ambiguous reference criteria; `medium-004` has uncertain evidence annotations and no gold passages. They remain included in the aggregates.

Verify the published aggregates without providers or model downloads:

```bash
.venv/bin/python -m evaluation.verify_published_results
```

## Interpretation and limitations

On this corpus, Hybrid and Hybrid + rerank retrieved relevant evidence more reliably than BM25 or Dense alone. Reranking had the highest test nDCG and passage coverage, with a CPU latency cost. These are measurements on one 56-question D&D 5e dataset, not universal claims about RAG quality.

The standalone and RAG answer outputs were reviewed by one LLM judge for correctness and completeness, but the results are exploratory rather than independently validated. Llama's partial CPU offload and the small GPU constrain latency. Better retrieval does not guarantee correct generation, and the evaluation does not isolate the causal contribution of reranking or the generation model.

## Reproduction

The saved run manifests contain the effective configuration and fingerprints. To reproduce a new run, start the existing Ollama service with the persistent model volume, then use the documented commands in [docs/datapizza-benchmark.md](../docs/datapizza-benchmark.md). Resume an interrupted run only with its exact run directory and matching configuration fingerprint.
