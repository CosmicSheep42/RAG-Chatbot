# Benchmark readiness

## Decision: NOT_READY

Do not run the full benchmark. The latest 12-question GPU pilot completed, but several generated answers still contain factual substitutions or unsupported comparisons. In particular, `easy-006`, `easy-012`, `easy-018`, `easy-019`, and `medium-014` have identifiable errors or omissions. No refusal-rate target overrides this evidence.

## Tested state

- Base Git revision: `7a36b56020ee4f98b3b1cd4726fea952b21b10c6`.
- Working tree at inference: dirty; source fingerprint: `3324197f1a3d06fe982866468db33f271701e3b668d0ffa3e4f918f28393f37a` in the run manifest.
- Final pilot: [`run-20261113T034432Z`](pilots/full-rag-gpu-dev/run-20261113T034432Z/manifest.json), 12 development IDs exactly once; no held-out questions.
- Dataset revision: `f2428dce66ea418a785998fa6ec57bc561b72239`.
- Runtime: Python 3.12.16; NVIDIA RTX 3050 Laptop GPU, 4096 MiB; driver `550.163.01`; Ollama `0.32.1`.
- Models/configuration: `llama3.2:3b`, `nomic-embed-text`, temperature 0, 500-token chunks, 100 overlap, top-K 5, one concurrent inference, no reranker.
- GPU use confirmed during inference by `nvidia-smi`: Ollama's `llama-server` used about 1.9 GiB; measured GPU utilization was 48% (a second pilot observation peaked at 95%).

## Validation

- Targeted reliability tests: **14 passed**.
- Ruff lint: passed.
- Ruff format: the edited answer service and focused test passed. A check including `settings.py` and `context_builder.py` still reports pre-existing formatting differences elsewhere in those files; no broad formatting rewrite was applied.
- Dataset validation and Docker Compose configuration: passed.
- Pilot consistency: 12/12 completed, exact expected ID set once each, and summary metrics recomputed from records.
- Full automated suite: **136 passed** in the pilot Docker Python 3.12.16 environment. The host Python 3.12.13 run timed out; the full container run is the authoritative result for this code state.
- Existing single-question GPU smoke: completed on `datapizza-medium-014`; `top_k="10"` normalized and returned six documents. Its answer repeated the known grappler/target confusion, supporting the `NOT_READY` decision. Output was written to disposable `/tmp`, leaving historical smoke files unchanged.

## Pilot metrics

- Completion: 12/12; failed executions: 0.
- Explicit refusals: 0; textual refusal heuristic: 0. These counts do not represent answer correctness.
- Tool calls: 1; valid: 1; validation/execution errors: 0/0.
- Searches: one duplicate-only result; no new documents or selected recovered evidence; successful evidence recoveries: 0.
- Citation identifiers: 8; invalid identifiers: 0. This checks identifier resolution only, not claim support.
- Incomplete responses: 1.
- Latency: p50 11.00 s, p95 20.45 s, max 20.67 s.

## Remaining blockers

1. The 3B generator still confuses facts across selected passages. The prompt is now delivered as a system message and the context budget matches the model's 4096-token window, but this does not guarantee grounding.
2. `easy-006` confuses Sorcerer and Warlock recovery features; `easy-012` limits Mass Heal to six targets; `easy-018` invents a damage feature name; `easy-019` misstates Sight Rot's affected creatures/transmission; `medium-014` assigns the escape check to the grappler.
3. `easy-009`, `medium-015`, `medium-020`, and `medium-023` remain incomplete or overly indirect. `medium-016` now declines the unsupported damage calculation, which is safer but still does not answer the question.
4. The full test suite did not finish in this environment.

The earlier exposed held-out `datapizza-easy-000` remains diagnostic-only and was not replayed or tuned against. Historical published directories, corpus, dataset annotations, and official benchmark results were not changed. No full benchmark, held-out evaluation, or judge run was performed.
