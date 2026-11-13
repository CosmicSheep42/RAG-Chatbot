# Pre-benchmark reliability review

## Scope and integrity

Reviewed source at base revision `7da0025966180ebd178680630fc6526c0a588c98` plus the dirty working-tree changes listed in the canonical pilot manifest. The final pilot source fingerprint is `ab53556dcffd668ea4dc230e324f322b99cf25c143f92f0e5e8597ba91bb6411`; the final refusal-classifier postprocessing fingerprint is recorded separately in that manifest. The resulting source state is not a clean Git revision.

The diagnostic case `datapizza-easy-000` belongs to the held-out split and is already exposed. It was not replayed or used for tuning in this pass. Only the twelve development questions were used in the new pilot. Dataset files, corpus, annotations, published benchmark directories and historical judge scores were left unchanged.

## Confirmed issues and changes

### Coverage and confidence semantics

`CoverageReport.score` measures corpus-relative structural distribution. It is not a calibrated probability of semantic relevance or question completeness and is not comparable as a universal threshold across intents. The confidence estimator previously treated this score as such by requiring an intent profile minimum (about `0.45`) even when the coverage evaluator declared the selected evidence structurally sufficient. That made its `INSUFFICIENT_COVERAGE` result contradict `CoverageReport.sufficient`.

The change removes that numeric score floor from confidence decisions. The decision uses the coverage evaluator's structural sufficiency result and emits `INSUFFICIENT_STRUCTURAL_COVERAGE` when that gate fails; the raw structural score remains diagnostic. Comparison coverage also requires evidence across distinct semantic units or adequate intent-aware unit coverage, so duplicated chunks and structural/neighbor filler cannot alone satisfy it. This aligns the two components' structural-coverage meaning, but does not establish semantic completeness. The development findings below show this is still not a sufficiently strong gate for the complete answer flow.

The prompt passed to answer selection now contains the original question. Answer prompting explicitly constrains factual claims to selected evidence, requests disclosure of missing requested details, and asks for citations. It does not run a second model check and therefore cannot guarantee grounding.

### Recovery and evidence accounting

Tool searches are normalized before equivalent-query comparison; repeated equivalent searches can be suppressed. If recovery introduces no new documents, the pipeline skips a redundant confidence reassessment. Pilot instrumentation distinguishes calls, new documents, duplicate-only searches and newly recovered evidence selected into final context. Existing bounds, tool validation and `top_k` normalization remain in place.

These are targeted latency and observability corrections. In the latest pilot the single tool call was duplicate-only, so the new evidence-selected count was zero. There is not enough recovery activity in twelve questions to claim a general recovery effectiveness improvement.

## Development case review

Qualitative review compared each stored answer with its reference information and the actual final selected context. Gold passage overlap is a diagnostic signal only. No LLM judge was used, and no accuracy score is claimed.

| Case | Classification | Finding |
|---|---|---|
| `easy-012` | Generation failure | The new answer has sufficient topical context but reverses two Mass Heal/Mass Healing Word target-count facts. Selected evidence did not prevent a factual mistake. |
| `easy-019` | Generation failure / incomplete support | The response generalizes Humanoid-only applicability to both diseases and applies the 1d4-day onset to both durations. Relevant facts were available, but the answer overgeneralized them. |
| `easy-016` | Insufficient evidence; model safety refusal | Selected context lacks the needed defensive feature details. The engine does not mark it as an explicit RAG refusal, but the answer begins with an unrelated harmful/illegal-content refusal. This is a generation behavior failure after retrieval failed; a factual answer is not supported. |
| `medium-016` | Generation failure | The answer confuses Acid Arrow, Eldritch Smite and Animate Objects, then makes an unsupported level/slot calculation despite having selected context. |
| `medium-023` | Incomplete answer | The response marks its list incomplete but adds unsupported ability-score/multiclass suggestions. |
| `easy-018` | Partially supported / generation failure | Main feature details are in context, but the answer invents a feature label and includes a nonsensical claim about effectiveness against allies and enemies. |
| `medium-011` | Appears supported | The stated strength value is supported by selected evidence in the final run. This is a qualitative inspection, not an accuracy metric. |
| `medium-020` | Incomplete, partly inferential | The answer acknowledges missing rules but extrapolates GM discretion from a general simultaneous-effects rule. |
| `easy-006` | Incomplete / conservative | Some slot evidence is selected; the answer acknowledges missing comparative information. |
| `easy-009` | Partial answer | Relevant components are present, but requested V/S/M requirements are not fully established. Refusal wording heuristic had a known quoted-text false positive in an earlier run; final heuristic was anchored at answer start. |
| `medium-014` | Mostly supported with irrelevant claim | The core answer appears supported but adds an unrelated Mimic pseudopod statement. |
| `medium-015` | Incomplete / inferential | It acknowledges no direct Counterspell evidence and speculates about GM discretion and Wish. |

The `easy-012` and `easy-019` refusal symptoms from the earlier pilot changed to answers, but those answers contain material factual errors. The observed reduction in explicit refusal count is therefore not a quality improvement. `medium-016` remains an unsupported answer. The final pilot confirms unresolved grounding defects.

## Tests and status

Added deterministic coverage for structural coverage semantics, missing comparison evidence, selected-context confidence, recovery deduplication/reassessment, original-question prompting, refusal classification and existing tool argument normalization. The complete test suite passed **133 tests** in the benchmark Python environment earlier in this work. The final host rerun collected all 133 tests but stalled in `tests/integration/test_engine.py::test_engine_composes_routing_retrieval_generation_and_citations` until its 60-second timeout; the benchmark image does not include pytest, so that rerun could not be repeated there. Ruff lint and format checks, dataset validation and Docker Compose validation passed. No GPU is needed by unit tests.

**Readiness implication:** not ready for the complete benchmark. The numeric coverage contradiction is removed, but structural sufficiency is not semantic answer completeness, and the generation prompt does not stop factual substitutions or unsupported additions. A targeted, general answer-support control or policy correction must be demonstrated on development and synthetic cases before full benchmarking. Do not tune thresholds against the exposed held-out `easy-000` case.
