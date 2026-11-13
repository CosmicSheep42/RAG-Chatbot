# Full production RAG GPU development pilot

## Run identity

Canonical run: [`run-20261113T003857Z` manifest](run-20261113T003857Z/manifest.json), with [records](run-20261113T003857Z/records.jsonl) and [summary](run-20261113T003857Z/summary.json). It used base revision `ceb42f26e60e5ded5401981995c284322761b210` plus the uncommitted working-tree changes in this workspace. The execution container could not query Git status; the manifest marks this unavailable. Dataset revision: `f2428dce66ea418a785998fa6ec57bc561b72239`. Exactly the 12 Datapizza development IDs listed in `manifest.json` were executed once. No held-out split, full benchmark matrix, or LLM judge was run.

The pilot used ChatbotEngine's Full production RAG path, `llama3.2:3b`, `nomic-embed-text`, temperature 0, chunk size 500, overlap 100, top-K 5, and one concurrent inference. Ollama image: `ollama/ollama:0.32.1`; persistent benchmark Ollama and data volumes were reused. RTX 3050 Laptop GPU (4096 MiB; NVIDIA driver `550.163.01`) was visible in-container using `nvidia-smi`. During generation, `ollama ps` reported `llama3.2:3b` at 100% GPU with 2.6 GB model memory; `nvidia-smi` showed about 2541 MiB allocated. Existing model digests were reused; no model download or reranker installation occurred. Reranker was unavailable.

## Aggregate results

| Measure | Result |
|---|---:|
| Completed / failed | 12/12 / 0 |
| Explicit refusals / textual refusal heuristic | 4/12 / 4/12 |
| Tool events / executed valid calls | 12 / 9 |
| Validation failures / execution failures | 0 / 0 |
| Searches returning documents / empty searches | 9 / 0 |
| Successful recovery searches / questions with relevant recovered evidence | 2 / 2 |
| Questions requiring recovery | 5/12 (41.7%) |
| Additional evidence documents | 17 |
| Citation identifiers / invalid identifiers | 10 / 0 |
| Incomplete responses flagged | 1 |
| End-to-end latency p50 / p95 / max | 7.37 s / 12.67 s / 14.95 s |

“Successful recovery” counts searches that introduced evidence; duplicate-only searches are not successful recoveries. One tool event per question on average includes three bounded/skipped model calls, so actual valid searches numbered nine. Citation validity is identifier consistency, not proof of factual correctness. The built-in assessment labels generated-answer correctness “Cannot determine automatically”; no factual score is claimed.

## Question review

The statuses below are a compact review against reference information and the stored selected evidence, not an LLM-judge score. “Appears supported” is a qualitative inspection, not verified correctness.

| Question | Outcome | Evidence / quality note |
|---|---|---|
| easy-006 | Refused | Some spell-slot evidence retrieved, but incomplete comparison; refusal is conservative. |
| easy-009 | Answered | Partial coverage of the asked V/S/M requirements; valid citations, completeness uncertain. |
| easy-012 | Refused | All 3 gold passages overlap selected context; likely coverage-policy false refusal. |
| easy-016 | Refused | 0/8 gold overlap after recovery; refusal justified by missing evidence. |
| easy-018 | Answered | Contains a likely unsupported feature label (“Create Explosive Elemental Effects”); review required. |
| easy-019 | Refused | 2/2 gold overlaps, but trace flags relevant evidence removed during selection; likely policy/context inconsistency. |
| medium-011 | Answered | Strength claim appears unsupported / potentially incorrect. |
| medium-014 | Answered | Appears correct and supported by selected source. |
| medium-015 | Answered | Cautious but incomplete on the Counterspell/Wish interaction. |
| medium-016 | Answered | 0/3 gold overlap; potentially unsupported Acid Arrow advice. |
| medium-020 | Answered | Incomplete and partly inferential. |
| medium-023 | Answered, partial | Only 3/24 gold-span overlaps; list is incomplete. |

## Reliability observations

The final code preserves recovered documents for reassessment, distinguishes new evidence from duplicates, records tool outcomes even on refusal, and bounds model-issued recovery searches. Still, easy-012 shows the coverage report can say evidence is sufficient while the confidence profile refuses; easy-019 flags evidence removal; and several generated answers warrant support/completeness review. Recovery sometimes spends extra time on duplicate-only fallback searches. These make the pilot useful as a diagnostic, but not a clean readiness signal for a full benchmark.

Run history is retained without overwriting: `run-20261113T003805Z` is an infrastructure failure (Ollama was stopped; 0 completed); earlier November 12 runs are superseded intermediate attempts. Only `run-20261113T003857Z` is the final 12/12 record set. The separate GPU smoke runner still passed after the pilot (`evaluation/smoke/gpu/runner-post-pilot.json`). Historical published benchmark files remain unchanged.
