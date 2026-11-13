# Pre-benchmark GPU development pilot comparison

## Runs and configuration

Baseline: [`run-20261113T003857Z` manifest](run-20261113T003857Z/manifest.json), with its [records](run-20261113T003857Z/records.jsonl) and [summary](run-20261113T003857Z/summary.json), at base revision `ceb42f26e60e5ded5401981995c284322761b210` plus the then-dirty source tree.

New canonical run: [`run-20261113T024514Z`](run-20261113T024514Z/manifest.json), [`records`](run-20261113T024514Z/records.jsonl), [`summary`](run-20261113T024514Z/summary.json). It used base revision `7da0025966180ebd178680630fc6526c0a588c98`, working-tree fingerprint `ab53556dcffd668ea4dc230e324f322b99cf25c143f92f0e5e8597ba91bb6411`, and the twelve development IDs in its manifest exactly once. Answers were not modified after inference; diagnostic refusal classification was recomputed after the classifier correction.

Both runs used production `ChatbotEngine` with `llama3.2:3b`, `nomic-embed-text`, temperature 0, 500-token chunks, 100-token overlap, top-K 5, and one concurrent inference. Dataset revision was `f2428dce66ea418a785998fa6ec57bc561b72239`. GPU was the NVIDIA RTX 3050 Laptop GPU (4096 MiB, driver `550.163.01`), Ollama image `0.32.1`. Runtime diagnostics observed generation and embedding at 100% GPU. Existing model/data volumes were reused; no reranker or model download was used.

## Aggregate comparison

| Metric | Baseline | New | Change |
|---|---:|---:|---:|
| Completed | 12/12 | 12/12 | — |
| Engine explicit refusals | 4 | 0 | -4 |
| Textual refusal heuristic | 4 | 1 | -3 |
| Tool calls / valid calls | 12 / 9 | 1 / 1 | -11 / -8 |
| Tool validation / execution failures | 0 / 0 | 0 / 0 | unchanged |
| Empty searches | 0 | 0 | unchanged |
| Duplicate-only searches | not separately summarized | 1 | observed |
| Recovery searches adding docs / selected new evidence | 2 / 17 docs previously counted as additional evidence | 0 / 0 | less retrieval; no selected recovery evidence |
| Citation identifiers / invalid identifiers | 10 / 0 | 15 / 0 | +5 / unchanged |
| Incomplete responses flagged | 1 | 1 | unchanged |
| Latency p50 | 7.37 s | 8.90 s | +1.53 s |
| Latency p95 | 12.67 s | 14.57 s | +1.90 s |
| Maximum latency | 14.95 s | 16.13 s | +1.18 s |

New recovery counters deliberately distinguish documents added to the candidate set from useful evidence selected in final context. The prior `17 additional evidence documents` aggregate is not directly comparable to the new stricter selected-evidence metric. The new run had one duplicate-only search, zero new documents, and zero new evidence selected. Citation validity means identifiers resolve to selected sources; it does not establish claim support.

## Per-question comparison

“Selected” is final context document count. Qualitative assessments are manual inspection against reference information and selected context, not a judge score.

| Question | Explicit refusal old → new | Selected old → new | Latency old → new (s) | New answer review |
|---|---|---:|---:|---|
| easy-006 | Yes → No | 19 → 12 | 9.91 → 9.06 | Incomplete; acknowledges missing comparative evidence. |
| easy-009 | No → No | 5 → 5 | 5.45 → 9.64 | Partial; requirements not fully established. |
| easy-012 | Yes → No | 15 → 12 | 10.81 → 8.66 | Factual substitutions in Mass Heal target counts. |
| easy-016 | Yes → No* | 12 → 12 | 7.14 → 6.58 | No required evidence; model gives unrelated safety refusal. `*` Text classifier flags it; engine refusal flag does not. |
| easy-018 | No → No | 5 → 5 | 5.47 → 8.28 | Partially supported; invented label and erroneous claim. |
| easy-019 | Yes → No | 12 → 10 | 8.42 → 13.29 | Overgeneralizes disease applicability and duration. |
| medium-011 | No → No | 5 → 5 | 4.73 → 6.16 | Appears supported. |
| medium-014 | No → No | 5 → 5 | 6.33 → 7.43 | Core appears supported; irrelevant Mimic detail. |
| medium-015 | No → No | 5 → 5 | 6.64 → 9.46 | Incomplete; inferential GM/Wish discussion. |
| medium-016 | No → No | 5 → 5 | 8.76 → 8.74 | Unsupported damage/slot claims persist. |
| medium-020 | No → No | 8 → 8 | 7.60 → 9.43 | Incomplete and partly inferential. |
| medium-023 | No → No | 15 → 24 | 14.95 → 16.13 | Explicitly incomplete; adds unsupported suggestions. |

## Interpretation and readiness

The new code eliminates the numeric structural-score contradiction and reduces repeated tool work. It also changes three prior engine refusals to answers (`easy-006`, `easy-012`, `easy-019`); at least two of those new answers contain clear factual defects. The remaining `easy-016` output is a safety-style refusal from the model despite absent evidence and an engine status of not-refused. `medium-016` continues to produce an unsupported answer. These results show fewer engine refusals do not demonstrate improved correctness.

No LLM judge was used. The answer review is a compact qualitative inspection, not a verified accuracy metric. The 12-question pilot completed on GPU with no tool failures and valid citation identifiers, but grounding/completeness remains a blocker. **Do not run the full benchmark yet.**
