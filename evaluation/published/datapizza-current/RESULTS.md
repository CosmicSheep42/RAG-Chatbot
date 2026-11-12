# Datapizza RAG Benchmark Results

## Current execution (2026-11-12)

The current run used the original 20-document corpus and 56-question dataset (12 development, 44 held-out), with chunk size 500, overlap 100, top-K 5, Chroma namespace `rag`, temperature 0, `llama3.2:3b`, and `nomic-embed-text`. The benchmark ran sequentially in Docker on Debian 13, Python 3.12.16, an Intel Core i5-12500H (16 logical CPUs), and 15.3 GiB RAM. No NVIDIA driver/GPU was available. Ollama was 0.40.16; Docker Engine was 26.1.5. This differs from the historical RTX 3050/4 GB and Ollama 0.32.1 setup, so latency comparisons are descriptive only.

The matrix had 20 cells. Sixteen completed with real Ollama inference: BM25, Dense, Hybrid, Standalone, and Full across their supported splits/modes. The four Hybrid + Rerank cells were attempted but could not initialize because `sentence-transformers` was absent. Its installation build was stopped after sustained 50–80 KB/s package downloads; no alternate model was used. The full production pipeline ran with reranking mode `auto`, but without the optional cross-encoder. The selected publication contains 448 successful question executions (168 retrieval and 280 generation), with zero failed questions. Six retrieval runs and one Full development run were repeated to capture signal/trace fields; including these duplicates, 628 successful question executions are present in the raw output tree and excluded duplicates do not affect published aggregates. The matrix execution window was 2 hours 21 minutes 50 seconds. Rerank cells have zero question records and are listed as unavailable, not as measured failures.

The full invocation and environment record is [environment.json](published/datapizza-current/environment.json); the four unavailable attempts are in [blocked-scenarios.json](published/datapizza-current/blocked-scenarios.json). The selected run IDs and complete records are in [published/datapizza-current/](./). Unselected duplicate/retry outputs remain under the ignored local results root and were not mixed into aggregates. The immutable earlier publication remains [published/datapizza/](../datapizza/).

## Retrieval comparison: historical versus current

Current measurements use the same dataset revision and annotations as the historical run. Test passage metrics use 43 questions with gold passage annotations; one question has no gold passages. Differences are current minus historical. Small changes are descriptive, not statistical claims. Current rerank is unavailable. Current timings are CPU-only and must not be compared as hardware-equivalent to the historical GPU timings.

| Split / measure | BM25 historical → current | Dense historical → current | Hybrid historical → current | Hybrid + rerank historical → current |
|---|---:|---:|---:|---:|
| Dev Recall@5 | 0.5741 → 0.5741 (0.0000) | 0.5440 → 0.6273 (+0.0833) | 0.6227 → 0.6227 (0.0000) | 0.7199 → N/M |
| Dev MRR@5 | 0.5694 → 0.5694 (0.0000) | 0.7222 → 0.8056 (+0.0834) | 0.7917 → 0.7778 (-0.0139) | 0.8361 → N/M |
| Dev nDCG@5 | 0.5108 → 0.5108 (0.0000) | 0.5650 → 0.6483 (+0.0833) | 0.6363 → 0.6312 (-0.0051) | 0.7062 → N/M |
| Dev weighted passage coverage | 0.2726 → 0.2726 (0.0000) | 0.2858 → 0.3483 (+0.0625) | 0.2963 → 0.3050 (+0.0087) | 0.2994 → N/M |
| Dev complete passage evidence | N/M → 0.1667 | N/M → 0.1667 | N/M → 0.0833 | N/M → N/M |
| Test Recall@5 | 0.6936 → 0.6936 (0.0000) | 0.6605 → 0.6760 (+0.0155) | 0.7446 → 0.7311 (-0.0135) | 0.7576 → N/M |
| Test MRR@5 | 0.7984 → 0.7984 (0.0000) | 0.7907 → 0.8101 (+0.0194) | 0.8624 → 0.8624 (0.0000) | 0.8605 → N/M |
| Test nDCG@5 | 0.6734 → 0.6734 (0.0000) | 0.6655 → 0.6793 (+0.0138) | 0.7360 → 0.7267 (-0.0093) | 0.7592 → N/M |
| Test weighted passage coverage | 0.3002 → 0.3002 (0.0000) | 0.2472 → 0.2937 (+0.0465) | 0.3298 → 0.3314 (+0.0016) | 0.4017 → N/M |
| Test complete passage evidence | 0.1163 → 0.1163 (0.0000) | 0.1163 → 0.1628 (+0.0465) | 0.1163 → 0.1163 (0.0000) | 0.2093 → N/M |

Recall, MRR, nDCG, passage recall, weighted coverage, and complete-evidence rate are recomputed from the selected per-question records. BM25 is unchanged at the reported precision. Dense improves on the held-out retrieval measures shown, while Hybrid is effectively unchanged on MRR/coverage and slightly lower on Recall/nDCG. No conclusion is possible for the reranker. Retrieval p50/p95 are included per run in the records; cross-machine changes are not meaningful because the historical host used GPU offload and the current host did not.

| Split / retrieval p50 / p95 | BM25 historical → current | Dense historical → current | Hybrid historical → current |
|---|---:|---:|---:|
| Dev | 28.0 / 30.3 ms → 12.8 / 15.4 ms | 29.2 / 33.5 ms → 40.3 / 56.7 ms | 79.4 / 102.7 ms → 119.3 / 178.7 ms |
| Test | 31.3 / 44.4 ms → 14.9 / 27.0 ms | 35.7 / 50.7 ms → 39.4 / 72.6 ms | 89.8 / 128.1 ms → 131.4 / 219.2 ms |

The latency pairs are observations only because the hardware/runtime differs; they are not speedup claims.

## End-to-end held-out results

All measured current configurations completed 44/44 questions. The benchmark marks all questions answerable. Current total p50/p95 seconds and the existing deterministic refusal heuristic are shown below; the heuristic is not human judgment. Historical latency is shown for context but is not a like-for-like comparison. Historical answer-quality scores below belong only to historical answers.

| Configuration | Historical total p50 / p95 s | Current total p50 / p95 s | Current generation p50 s | Current refusal heuristic | Current citation IDs |
|---|---:|---:|---:|---:|---:|
| Standalone | 4.12 / 12.09 | 5.99 / 18.27 | 5.99 | 27/44 (61.4%) | none emitted |
| BM25 RAG | 7.48 / 16.22 | 23.14 / 38.55 | 23.11 | 10/44 (22.7%) | 15 answers cite; 0 invalid IDs |
| Dense RAG | 14.62 / 23.16 | 26.22 / 43.34 | 26.10 | 14/44 (31.8%) | 18 answers cite; 0 invalid IDs |
| Hybrid RAG | 13.11 / 21.49 | 24.52 / 46.67 | 24.28 | 10/44 (22.7%) | 18 answers cite; 0 invalid IDs |
| Hybrid + rerank RAG | 12.34 / 18.01 | N/M | N/M | N/M | N/M |
| Full production RAG | 7.88 / 13.48 | 21.79 / 146.33 | not exposed | 18/44 heuristic; 9 explicit refusals | 14 answers cite; 0 invalid IDs |

The Full pipeline p95 includes one 146.33-second query. Component retrieval and generation timings are not exposed by the production API; the harness captures total engine time and full pipeline trace. Its citation identifier validity was 1.0 when IDs were emitted. Historical Full citation validity was also 1.0; automated source relevance/support proxy increased from 0.535 to 0.580, which does not establish factual correctness or groundedness.

The trace records confidence decisions, observed retrieval calls, query rewrites, and tool events. The production API does not expose internal retrieval retry events as a count, so `recovery_attempts` is null rather than inferred.

Full RAG explicitly refused 9 held-out items, all of which the dataset labels answerable. Across its 44 answers, the phrase-based heuristic flagged 18 as incorrect refusals; these are different measures. Nine tool calls failed before returning evidence because the generated `top_k` argument was either a string or greater than the tool's maximum of 6. The recorded calls produced zero results. Confidence values and decisions are diagnostic signals, not calibrated correctness probabilities.

The current blinded review package contains the new generated answers and reference/evidence context in `judge_input.jsonl`, with a matching empty rubric template. It has not been scored. The prior GPT-6 judge ratings in [evaluation/judge/datapizza/](../../judge/datapizza/) describe only historical answers and are not used as current quality results. See the detailed [baseline comparison](../../comparisons/baseline-vs-current/REPORT.md).

## Historical baseline and verification

The original GPU measurements, records, checksums, manifests, and judge annotations remain unchanged under [evaluation/published/datapizza/](../datapizza/). The current export is separate. Verify both record bundles offline with:

```bash
python3 -m evaluation.verify_published_results evaluation/published/datapizza
python3 -m evaluation.verify_published_results evaluation/published/datapizza-current
```

The dataset, corpus, answer text, and annotations were not changed. The benchmark harness was minimally instrumented to capture real Full-engine evidence, confidence, retrieval attempts, tool activity, and citation IDs; no production behavior was changed. See [the reproduction guide](../../../docs/datapizza-benchmark.md) for commands and limitations.
