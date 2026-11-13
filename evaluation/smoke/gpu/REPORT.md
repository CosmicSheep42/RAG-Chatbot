# GPU and recovery smoke report

This records two development question IDs, not a benchmark. `datapizza-medium-014` was rerun against the final code; `datapizza-easy-016` is preserved as an earlier attempt before the final empty-search guard. Neither updates published results or the baseline comparison. Raw outputs are in `results.json` and `earlier-refusal.json`; host/runtime measurements are in `environment.json`.

## Confirmed recovery defects

- The nine published `search_documents` errors are schema-validation failures before handler execution. Their `top_k` values were either numeric strings such as `"10"` or integers above the schema maximum of 6. The handler's former conversion and clamp could not run. `ToolRegistry` now normalizes only the explicitly registered `search_documents.top_k` field before strict validation, reads default/minimum/maximum from its schema, and rejects booleans, floats, malformed strings, and values below the minimum.
- `ToolRegistry` now reports `validation_error`, `execution_error`, `success_empty`, or `success`. It does not expose exception details. `search_documents` rejects empty query text, and its schema disallows unexpected properties.
- In `datapizza-easy-000`, the trace has confidence `0.873275` with decision `retry`, but the selected context is empty and the tool-call list is empty. This was not one of the `top_k` failures: recovery-only execution relied on the model to request a search and then reassessed an empty context. The bounded recovery path now invokes one registry-validated search using the standalone query if the model makes no search or its search call fails validation/execution. A successful empty result remains `success_empty`; it gets one fallback only when the original standalone query differs, avoiding an identical duplicate. Final confidence still controls refusal.

The production engine still owns final context selection, deduplication, confidence reassessment, refusal, and citation construction. A retrieval that returns documents does not itself grant answer permission. Focused deterministic tests exercise recovery with evidence, errors, empty results, sync/async calls, and refusal. The earlier `datapizza-easy-016` smoke attempt was refused and had no tool calls; it predates the final empty-search guard. No broad refusal or accuracy improvement is claimed.

## GPU verification

The host is Debian 13, not WSL. `lspci` detects the RTX 3050 Laptop GPU; the loaded NVIDIA driver is 550.163.01. An initial unprivileged diagnostic could not access the driver, but the approved Docker diagnostic exposed the GPU correctly. Docker has the NVIDIA runtime, the Compose GPU override requests one device, and the Ollama container received a GPU `DeviceRequest`.

The generation model returned a real response and `ollama ps` reported `100% GPU`. During generation, `nvidia-smi` observed 2,537 MiB used and up to 94% utilization. The short generation took 2.823 s including model load. The configured `OllamaEmbeddings` provider returned 768 finite dimensions in 1.984 s; Ollama reported `nomic-embed-text` at `100% GPU`, with 401 MiB observed afterward. No OOM occurred. The reranker was neither downloaded nor loaded.

The production `ChatbotEngine` loaded the persisted Chroma index without reindexing and answered `datapizza-medium-014` in 5.498 s with five citation identifiers whose sources were present. A separate call through its registered search handler with `top_k="10"` returned six documents successfully in 1.636 s. Afterward, `ollama ps` reported `llama3.2:3b` at `100% GPU` and `nvidia-smi` observed 2,541 MiB allocated. The assistant did not need model-issued tools for this answer; sync/async tool and recovery behavior is covered by deterministic tests.

## Remaining issues and limits

- `datapizza-easy-016` remained refused by the production engine, with no source or citation. This single smoke failure needs its retrieval/context trace inspected; this run does not establish its root cause.
- Model-generated tool argument normalization was validated at the production registry boundary, but the smoke questions did not demonstrate a real model-issued tool call.
- The short generation response itself was a functional connectivity check, not an answer-quality evaluation.
- No complete benchmark, held-out batch, or judge evaluation was run. CPU Compose syntax was checked, but CPU inference was not run.
- Chroma emitted telemetry compatibility warnings during engine initialization; the smoke completed. They did not prevent retrieval or generation.

## Automated validation

The complete automated suite passed in a disposable Python 3.12 container: 120 tests passed, 52.16% coverage (above the repository's 35% minimum). `ruff check --no-cache .` and `ruff format --check --no-cache evaluation tests` passed. The local uv-managed Python environment stalled on sequential `asyncio.to_thread` calls; using the container matched the working application runtime and avoided that environment-specific issue.
