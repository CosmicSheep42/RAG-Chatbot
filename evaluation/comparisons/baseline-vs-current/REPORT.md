# Historical baseline vs current RAG implementation

## Scope and comparability

The historical values come from the preserved `evaluation/published/datapizza/` bundle and its original `evaluation/RESULTS.md`. Current values come from the selected CPU-only runs in `evaluation/published/datapizza-current/`. Both use the same 20-document corpus, dataset revision `f2428dce66ea418a785998fa6ec57bc561b72239`, question IDs, gold passages, 500/100 chunking, top-K 5, temperature 0, `llama3.2:3b`, and `nomic-embed-text`.

The old run used Ollama 0.32.1 and an RTX 3050 Laptop GPU with partial LLM GPU offload. The new run used Ollama 0.40.16 on an Intel i5-12500H CPU because no NVIDIA driver was usable. Therefore latency deltas are not interpreted as implementation regressions or improvements. The reranker could not be initialized in the new environment; rerank comparisons are unmeasured. The Full pipeline did run through the current production engine with reranking mode `auto`, but its optional cross-encoder was unavailable.

## Retrieval, held-out test split

Values are historical → current; the delta is absolute and relative to the historical value. Current passage metrics use 43 annotated questions; one test question lacks gold passages. “No meaningful change” means the observed difference is zero or small on this 44-question sample, not that equivalence was statistically established.

| Configuration / metric | Historical | Current | Absolute delta | Relative delta | Reading |
|---|---:|---:|---:|---:|---|
| BM25 Recall@5 | 0.6936 | 0.6936 | +0.0000 | 0.0% | No change |
| BM25 MRR@5 | 0.7984 | 0.7984 | +0.0000 | 0.0% | No change |
| BM25 nDCG@5 | 0.6734 | 0.6734 | +0.0000 | 0.0% | No change |
| BM25 weighted passage coverage | 0.3002 | 0.3002 | +0.0000 | 0.0% | No change |
| BM25 complete evidence rate | 0.1163 | 0.1163 | +0.0000 | 0.0% | No change |
| Dense Recall@5 | 0.6605 | 0.6760 | +0.0155 | +2.3% | Improvement on this split |
| Dense MRR@5 | 0.7907 | 0.8101 | +0.0194 | +2.5% | Improvement on this split |
| Dense nDCG@5 | 0.6655 | 0.6793 | +0.0138 | +2.1% | Improvement on this split |
| Dense weighted passage coverage | 0.2472 | 0.2937 | +0.0465 | +18.8% | Improvement on this split |
| Dense complete evidence rate | 0.1163 | 0.1628 | +0.0465 | +40.0% | Improvement on this split |
| Hybrid Recall@5 | 0.7446 | 0.7311 | -0.0135 | -1.8% | Small regression |
| Hybrid MRR@5 | 0.8624 | 0.8624 | +0.0000 | 0.0% | No change |
| Hybrid nDCG@5 | 0.7360 | 0.7267 | -0.0093 | -1.3% | Small regression |
| Hybrid weighted passage coverage | 0.3298 | 0.3314 | +0.0016 | +0.5% | No meaningful change |
| Hybrid complete evidence rate | 0.1163 | 0.1163 | +0.0000 | 0.0% | No change |
| Hybrid + rerank | 0.7576 Recall / 0.7592 nDCG / 0.4017 coverage | Not measured | — | — | Dependency unavailable |

BM25 current Recall/MRR/nDCG/coverage exactly match the baseline on both splits at stored precision. Dense is higher on each held-out metric shown, although dev and held-out gains do not establish the cause. Hybrid remains near the baseline, with a slight test Recall/nDCG dip. Dense still has zero passage recall on `datapizza-medium-019`; Hybrid also has zero passage recall there. These examples show that aggregate gains do not remove query-level retrieval misses.

Retrieval latency p50/p95 was measured in each new run and is preserved in the records. The current held-out p50 values were 14.9 ms BM25, 39.4 ms Dense, and 131.4 ms Hybrid. Historical values were 31.3, 35.7, and 89.8 ms on the GPU-enabled host. The hardware and Ollama versions differ, so these are reported as observations, not comparable deltas.

| Split / retrieval latency | BM25 historical → current | Dense historical → current | Hybrid historical → current |
|---|---:|---:|---:|
| Dev p50 / p95 | 28.0 / 30.3 ms → 12.8 / 15.4 ms | 29.2 / 33.5 ms → 40.3 / 56.7 ms | 79.4 / 102.7 ms → 119.3 / 178.7 ms |
| Test p50 / p95 | 31.3 / 44.4 ms → 14.9 / 27.0 ms | 35.7 / 50.7 ms → 39.4 / 72.6 ms | 89.8 / 128.1 ms → 131.4 / 219.2 ms |

The p50/p95 values above are measured, but no change is labeled as a speed regression or improvement because the old host used GPU offload and the current host did not.

## End-to-end, held-out test split

All five available configurations completed 44/44 questions with zero execution failures. The rerank scenario did not execute questions. The table compares the same deterministic `is_abstention` phrase heuristic over historical and current answer text; every question is labeled answerable. It is not a correctness or human refusal evaluation.

| Configuration | Historical total p50/p95 s | Current total p50/p95 s | Latency status | Phrase-based refusal, old → current | Citation IDs in current answers |
|---|---:|---:|---|---:|---:|
| Standalone | 4.12 / 12.09 | 5.99 / 18.27 | Not comparable: CPU vs GPU | 26/44 → 27/44 (+1) | No citations expected |
| BM25 RAG | 7.48 / 16.22 | 23.14 / 38.55 | Not comparable: CPU vs GPU | 11/44 → 10/44 (-1) | 15 answers cited; 0 invalid IDs |
| Dense RAG | 14.62 / 23.16 | 26.22 / 43.34 | Not comparable: CPU vs GPU | 18/44 → 14/44 (-4) | 18 answers cited; 0 invalid IDs |
| Hybrid RAG | 13.11 / 21.49 | 24.52 / 46.67 | Not comparable: CPU vs GPU | 15/44 → 10/44 (-5) | 18 answers cited; 0 invalid IDs |
| Hybrid + rerank RAG | 12.34 / 18.01 | Not measured | Not comparable | Not measured | Not measured |
| Full production RAG | 7.88 / 13.48 | 21.79 / 146.33 | Not comparable: CPU vs GPU; current p95 includes a 146.33 s query | 17/44 → 18/44 (+1) | 14 answers cited; 0 invalid IDs |

The phrase heuristic fell for BM25, Dense, and Hybrid, was effectively flat for Standalone and Full, and does not show whether answers became more correct or complete. Current citations had no invalid IDs among emitted `[S#]` references. Valid identifiers do not establish that an answer is true or that each claim is supported. The historical Full source-level citation relevance proxy averaged 0.535 and the current proxy 0.580; this is not claim-level groundedness. Historical/current correctness and completeness are **not comparable** because no current judge evaluation was run. The separate current blinded review inputs and blank rubric are ready in the publication; historical GPT-6 annotations remain attached only to historical answers.

## Observed Full-pipeline behavior and failures

- The current Full engine explicitly marked 9/44 answerable test questions as refused. For `datapizza-easy-000`, confidence was 0.873 with decision `retry`, reason codes included `RERANKER_UNAVAILABLE` and `INSUFFICIENT_COVERAGE`, yet the final answer refused and selected no evidence. The score is an internal signal, not a calibrated probability.
- Nine recorded `search_documents` tool calls returned zero results because model-supplied `top_k` values violated the tool contract: string values such as `"10"` or integer 10 where the maximum is 6. Affected questions were `datapizza-easy-004`, `datapizza-medium-000`, `-006`, `-018`, `-021`, `-024`, `-025`, `-029`, and `-030`. This is direct evidence that the current tool recovery path can fail at argument validation.
- Retrieval still misses annotated evidence on individual questions, including `datapizza-medium-019` for Dense and Hybrid. `datapizza-easy-004` has partial-enumeration evidence in the Full trace; the generated response itself says it cannot establish one part of the requested effect. This is an observed incomplete answer candidate, not a judge-confirmed correctness label.
- Confidence reason codes report `RERANKER_UNAVAILABLE` on Full because the cross-encoder was not installed. The unavailable reranker is an environment limitation, not evidence that the reranker regressed.

## Three evidence-backed priorities

1. **Make tool arguments conform to the registered tool schema.** All nine observed Full tool calls failed validation and returned no evidence. Constrain `top_k` to an integer no greater than 6 and verify recovery returns evidence before treating it as successful.
2. **Review Full refusal decisions against answerable cases.** Nine held-out questions were explicitly refused; `easy-000` had strong multi-signal evidence and coverage marked sufficient but still ended in a refusal. Inspect the current confidence/retry transition with the captured evidence before tuning thresholds.
3. **Investigate query-level evidence misses and partial enumeration.** Dense/Hybrid both miss the gold passage for `medium-019`; `easy-004` is an observed partial response after retrieval/context selection. Use the stored ranked candidates, selected evidence offsets, and blinded review scores to separate retrieval misses from context-selection and generation failures.

## Reproducibility and limitations

The current selected publication contains 16 complete matrix cells and 448 successful question executions. Four reranker cells were attempted and stopped before question execution. The environment and run IDs are captured in `evaluation/published/datapizza-current/environment.json`, `selection.json`, and the per-run manifests. The initial repo tree was clean at commit `9d9031c830a7655abe6a27222ddbde4004821712`; benchmark instrumentation and focused tests were local working-tree changes, while production code, dataset, and corpus remained unchanged. Automated tests after instrumentation passed: 88 passed.

The current answers are not yet graded for correctness, completeness, or claim-level evidence support. This report makes no such quality claim. The dataset is a single small D&D 5e domain, and the refusal heuristic is only lexical. Reranker behavior and quality scores for the current generation are unmeasured. Historical runs and judge records remain preserved separately.

The engine trace exposes confidence decisions, retrieval calls observed by the adapter, query rewrites, and tool events, but the public production API has no explicit internal retry-event counter. Therefore `recovery_attempts` is recorded as null; retry decisions are not silently converted into counts.
