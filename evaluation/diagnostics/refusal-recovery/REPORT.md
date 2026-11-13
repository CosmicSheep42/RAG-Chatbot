# Refusal and recovery diagnosis

## Scope and integrity

The working tree was already dirty at the start of this task. The base Git revision is `ceb42f26e60e5ded5401981995c284322761b210`; the working source includes uncommitted changes, so that revision alone does not identify the tested code. The persisted pilot manifest records the base revision and notes that runtime Git status was unavailable inside the benchmark container.

`datapizza-easy-000` is in the held-out test split and has been exposed as a known diagnostic case. It was replayed only because it was explicitly named in the task. No other held-out questions were inspected, no gold annotations were changed, and no benchmark-wide tuning was performed. `datapizza-easy-016` is in the development split. Neither case is evidence of unbiased future test performance.

## `datapizza-easy-000`

The historical trace and the final-code replay both show this was not an initial context-selection wipeout. The rewritten question retrieved the Pit Fiend and Rakshasa restoration passages and the Solar passage, but not the Planetar restoration passage. Initial context selection kept all 15 input documents, including relevant evidence. Confidence was about `0.873` with `RETRY`; the coverage evaluator nevertheless reported `sufficient=true` and `coverage_ok` (score `0.2647` in the original trace).

The recovery search ran and returned six documents, including the Planetar passage (`09bc9d8c8fddb450`). The final preferred-evidence context selection kept all 17 documents and retained the exact passages for Pit Fiend, Planetar, Rakshasa, and Solar. A subsequent original-query search returned four documents, all duplicates, and added no new evidence. Tool execution itself succeeded. Despite the four-way evidence being present, final confidence remained about `0.873`, decision `RETRY`, with `INSUFFICIENT_COVERAGE`; its numeric coverage was `0.2774` while the coverage report said `sufficient=true` / `coverage_ok`. The final refusal therefore arose from conflicting coverage gates, not from evidence being lost or failed recovery.

**Root cause:** the confidence estimator applies a separate profile minimum coverage score (`0.45`) even when the intent-aware coverage report marks the selected evidence sufficient. This produces a false refusal for this trace. The refusal response also strips source references from its public response object; the stored context-selection trace, rather than `response.sources`, is needed to inspect evidence.

**Correction:** recovered documents are prioritized during reassessment/context assembly, preventing expansion filler from displacing newly found evidence. A bounded standalone fallback search now runs when model recovery is skipped, invalid, empty, duplicate-only, or still needs evidence; duplicate-only recovery is not counted as successful. One model-issued search is bounded per recovery response. Refusal traces retain tool observations. These changes improve evidence availability and observability, but do not eliminate the coverage-policy conflict.

## `datapizza-easy-016`

This development question was replayed in the final 12-question pilot. Initial retrieval and selected context each contained 12 documents; no relevant item was removed by context selection. The query did not retrieve the required defensive-reference / gold passages (0 of 8 gold passages overlapped the final selected set). Confidence was about `0.870`, but coverage was inadequate. Model recovery and the standalone fallback both returned documents but added no new documents relevant to the question. There were no tool validation or execution failures. The answer was refused.

**Root cause:** retrieval mismatch / missing supporting evidence. The high candidate confidence did not imply that the required facts were present. Recovery did not find useful new evidence, so the refusal was justified. No confidence threshold was lowered and no forced answer was introduced.

## Tests and remaining issues

Added deterministic regression coverage for recovered-evidence preservation, fallback after skipped/invalid/empty/duplicate-only recovery, bounded recovery calls, final refusal tool traces, confidence candidate ordering, and refusal when coverage remains insufficient. Synchronous/asynchronous recovery and existing `top_k` normalization paths are covered by the targeted suite.

The benchmark-container test run passed **127 tests**. The score-versus-coverage-report inconsistency remains unresolved: removing the numeric floor caused an evidence-poor case to be answered, so a safe general correction needs a better-calibrated policy and independent development fixtures. Fallback searches can also add latency when they return duplicates. Refusal behavior should be re-evaluated on a fresh untouched test set after policy work.
