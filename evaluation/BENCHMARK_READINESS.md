# Benchmark readiness

## Decision: NOT_READY

The production pipeline completed the controlled twelve-question development pilot on the RTX 3050, with zero tool validation/execution failures and syntactically valid citations. However, the latest answers include confirmed unsupported or factually incorrect claims, including `easy-012`, `easy-019`, and `medium-016`. A lower refusal count is not sufficient evidence of reliability. Do not run the full benchmark until answer grounding and completeness are corrected and revalidated.

## Tested state

- Base Git revision: `7da0025966180ebd178680630fc6526c0a588c98`.
- Working tree: dirty; source fingerprint recorded in canonical run manifest as `ab53556dcffd668ea4dc230e324f322b99cf25c143f92f0e5e8597ba91bb6411`.
- Canonical development pilot: `evaluation/pilots/full-rag-gpu-dev/run-20261113T024514Z/`.
- Pilot dataset revision: `f2428dce66ea418a785998fa6ec57bc561b72239`; exactly 12 dev IDs, once each; no held-out split.
- Runtime: Python 3.12.16; RTX 3050 Laptop GPU (4096 MiB), driver `550.163.01`; Ollama `0.32.1`.
- Models/configuration: `llama3.2:3b`, `nomic-embed-text`, temperature 0, chunk size 500, overlap 100, top-K 5, one concurrent inference; no reranker.
- GPU confirmed by `ollama ps` reporting generation and embeddings at 100% GPU and by `nvidia-smi` memory/utilization diagnostics.

## Validation completed

- Full automated suite: **133 passed** in the benchmark Python environment earlier in this work. A final host rerun collected the same 133 tests but stalled in `tests/integration/test_engine.py::test_engine_composes_routing_retrieval_generation_and_citations` until a 60-second timeout; the benchmark image lacks pytest, preventing a same-container rerun. This environment-specific stall remains a validation limitation.
- Ruff lint: passed.
- Ruff format check: passed (50 files).
- Datapizza dataset validation: passed.
- Docker Compose configuration validation: passed.
- Pilot: 12/12 completed, zero execution failures, zero tool validation/execution failures, 15 citation identifiers with zero invalid identifiers.
- Pilot latency: p50 8.90 s, p95 14.57 s, max 16.13 s.
- Pilot output consistency: final run has exactly the manifest's twelve development IDs. Historical published directories and dataset files have no working-tree changes.

## Remaining blockers

1. Structural coverage sufficiency is not semantic completeness. The numeric score contradiction was fixed, but the coverage gate can still allow answers when required facts are missing or evidence is not used correctly.
2. Answer generation introduced wrong target counts for `easy-012`, overgeneralized disease details for `easy-019`, and unsupported damage/slot claims for `medium-016`.
3. `easy-016` had insufficient evidence and emitted an unrelated safety refusal that was not represented by the engine's explicit-refusal flag.
4. `medium-023` and several others remain incomplete or partly inferential.
5. Recovery benefit is not demonstrated in the new pilot: one duplicate-only search, no newly selected evidence. This is not enough data to claim recovery reliability broadly.

The exposed held-out `datapizza-easy-000` remains diagnostic-only and was not used for tuning in this pass. No full benchmark, held-out evaluation, publication workflow, or LLM judge was run. A new untouched evaluation set should be considered for any future generalization claim.
