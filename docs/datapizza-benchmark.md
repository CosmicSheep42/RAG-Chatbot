# Datapizza Benchmark and Reproduction Guide

The Datapizza benchmark has been executed against the current implementation. This document is both the
reproduction guide and a record of the current run's limits. The historical GPU publication remains in
`evaluation/published/datapizza/`; current CPU results are separate in
`evaluation/published/datapizza-current/`. The prior judge scores in `evaluation/judge/datapizza/` apply
only to historical answers. Current blinded review input is included in the current publication and is
pending scoring.

The matrix attempted all 20 cells. Sixteen completed using the real local Ollama models; the four
Hybrid + Rerank cells stopped before question execution because `sentence-transformers` was absent. No
alternate model was substituted. Full used the production engine with `reranking_mode=auto`, but without
the optional cross-encoder. Environment, effective model identifiers, run IDs, checksums, and failed cells
are recorded in `evaluation/published/datapizza-current/`. Verify both publications independently:

```bash
python3 -m evaluation.verify_published_results evaluation/published/datapizza
python3 -m evaluation.verify_published_results evaluation/published/datapizza-current
```

The repository contains 20 Datapizza D&D 5e Markdown documents in `knowledge-base/` and 56 original QA
examples in `evaluation/dataset/datapizza/`. The split manifest contains 12 development and 44 test IDs.
The benchmark adapter preserves the upstream passage text and character offsets. One upstream medium
example has no passages; it remains preserved and passage metrics are not applicable for that example.

Validate the copied data without starting any service:

```bash
python evaluation/validate_dataset.py
.venv/bin/python -m evaluation.preflight --configuration bm25 --mode retrieval-only
```

The preflight is read-only apart from creating a temporary file in the output directory. It reports
runtime dependencies and Ollama connectivity as `UNAVAILABLE` or `NOT CHECKED`; it never starts a
service or downloads a model. Docker Compose configuration can also be inspected without starting it:

```bash
docker compose -f docker-compose.benchmark.yml config --quiet
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml config --quiet
```

Build the benchmark image only when preparing a machine for a future run. The build does not download an
LLM or execute a benchmark. Add `--build-arg INSTALL_RERANKER=true` when preparing the reranking scenario.

The default profile is sequential and uses `llama3.2:3b`, `nomic-embed-text`, Chroma, temperature `0`, and
one concurrent inference. Models must be prepared separately in the persistent Ollama volume:

```bash
docker compose -f docker-compose.benchmark.yml --profile standalone build
docker compose -f docker-compose.benchmark.yml --profile hybrid-rerank build --build-arg INSTALL_RERANKER=true
docker compose -f docker-compose.benchmark.yml --profile standalone up -d ollama
docker compose -f docker-compose.benchmark.yml --profile standalone exec ollama ollama pull llama3.2:3b
docker compose -f docker-compose.benchmark.yml --profile standalone exec ollama ollama pull nomic-embed-text
```

Run one scenario at a time. These commands execute new measurements; they are not needed to inspect or
verify the existing publications:

```bash
docker compose -f docker-compose.benchmark.yml --profile standalone run --rm benchmark-standalone
docker compose -f docker-compose.benchmark.yml --profile bm25 run --rm benchmark-bm25
docker compose -f docker-compose.benchmark.yml --profile dense run --rm benchmark-dense
docker compose -f docker-compose.benchmark.yml --profile hybrid run --rm benchmark-hybrid
docker compose -f docker-compose.benchmark.yml --profile hybrid-rerank run --rm benchmark-hybrid-rerank
docker compose -f docker-compose.benchmark.yml --profile full run --rm benchmark-full
```

The default mode is retrieval-only for BM25, Dense, Hybrid, and Hybrid + Rerank. Select end-to-end
generation without editing YAML:

```bash
docker compose -f docker-compose.benchmark.yml --profile bm25-e2e run --rm benchmark-bm25-e2e
BENCHMARK_MODE=end-to-end docker compose -f docker-compose.benchmark.yml --profile dense run --rm benchmark-dense
BENCHMARK_MODE=end-to-end docker compose -f docker-compose.benchmark.yml --profile hybrid run --rm benchmark-hybrid
BENCHMARK_MODE=end-to-end docker compose -f docker-compose.benchmark.yml --profile hybrid-rerank run --rm benchmark-hybrid-rerank
```

The BM25 retrieval-only service includes `--live` only to select the configured runner path; it constructs
only the in-memory BM25 index and does not create an LLM, embedding, Chroma, or vector index. The separate
`benchmark-bm25-e2e` profile adds the Ollama dependency and explicitly selects end-to-end generation while
still using only BM25 retrieval.

Standalone and Full are always end-to-end. Use `BENCHMARK_SPLIT=test` for the held-out split and
`BENCHMARK_TOP_K` or an explicit CLI override when running the image directly. BM25 retrieval-only does
not need Ollama, embeddings, Chroma, or paid APIs; generation modes require the configured model.
The result root is `evaluation/results/datapizza/<configuration>/<mode>/run-<id>/`, with a manifest,
incremental `records.jsonl`, and `summary.json`. A new run gets a unique directory and never overwrites
an existing one. Resume a specific incomplete run by invoking the CLI with `--resume --output <run-directory>`.
The manifest fingerprint checks dataset/corpus checksums, models, providers, chunking, retrieval limits,
reranking, prompt files, temperature, and mode. Completed runs are protected from overwrite:

```bash
python -m evaluation.benchmark \
  --live --configuration bm25 --mode end-to-end --split dev \
  --output evaluation/results/datapizza/bm25/end-to-end/run-<id> --resume
python -m evaluation.benchmark \
  --live --configuration bm25 --mode end-to-end --split dev \
  --output evaluation/results/datapizza/bm25/end-to-end/run-<id> --resume --retry-failed
```

Export generated answers for manual review and aggregate completed annotations. The published judge results
were produced separately by a blinded LLM-as-a-Judge workflow and must not be described as human review:

```bash
python -m evaluation.manual_review export <run-directory>/records.jsonl reviews.jsonl
python -m evaluation.manual_review aggregate reviews.jsonl
```

Review annotations must set `review_status` to `reviewed` and use scores 0, 1, 2, or null for correctness,
completeness, and evidence support. Unreviewed answers remain excluded from means.

## NVIDIA GPU preparation and smoke test

The base Compose file remains CPU-compatible. GPU use is explicit: add
`-f docker-compose.benchmark.gpu.yml` to select one NVIDIA device for the Ollama service. The reranker
remains independently configurable and defaults to CPU (`RERANKER_DEVICE=cpu`). Ollama is limited to
one parallel inference and one loaded model, which prevents the generation and embedding models from
competing for this 4 GB GPU.

The following checks and smoke were executed on Debian 13 with an RTX 3050 Laptop GPU, NVIDIA driver
550.163.01, Docker 26.1.5, Compose 2.26.1, and Ollama image `0.32.1`. Host ownership and driver installation
are prerequisites; do not install or alter host drivers as part of a benchmark run.

Check host driver access and both Compose configurations:

```bash
nvidia-smi
docker info --format '{{json .Runtimes}}'
docker compose version
docker compose -f docker-compose.benchmark.yml config --quiet
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml config --quiet
```

GPU availability, Docker visibility, and inference acceleration are separate checks. On this host, the
GPU container test below saw `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm`; `nvidia-smi` reported
4,096 MiB. During a generation, `nvidia-smi` observed 2,537 MiB allocated and up to 94% utilization, while
`ollama ps` reported `100% GPU`. These are the confirmations that inference actually used the GPU:

```bash
docker run --rm --gpus all --entrypoint /bin/sh ollama/ollama:0.32.1 -c 'ls -l /dev/nvidia*'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

Start only the benchmark Ollama service, using its persistent model volume. Pull the exact models only if
they are absent; the smoke does not pull or load the CPU reranker:

```bash
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full up -d ollama
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full exec -T ollama ollama list
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full exec -T ollama ollama pull llama3.2:3b
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full exec -T ollama ollama pull nomic-embed-text
```

The commands above were used here; the existing `benchmark-ollama` volume was reused. Run a short
generation, then inspect both the processor placement and device memory. `ollama ps` reports partial
offload as a CPU/GPU percentage; do not infer GPU use from container startup alone:

```bash
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full exec -T ollama ollama run --verbose llama3.2:3b 'Reply with one word: ready.'
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full exec -T ollama ollama ps
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

This short generation completed in 2.823 seconds including model load. The configured
`OllamaEmbeddings` provider returned a 768-dimensional vector in 1.984 seconds and was reported as
`100% GPU`. Its exact provider smoke command was:

```bash
docker run --rm --network rag-chatbot_default --entrypoint python rag-chatbot-benchmark:local -c 'from langchain_ollama import OllamaEmbeddings; values=OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434").embed_query("A short embedding smoke test."); print(len(values), all(abs(value)<float("inf") for value in values))'
```

The limited production-engine smoke uses only `datapizza-medium-014` from the development split and
stores its response separately from official results. The exact invocation validated here mounted the
current source read-only and reused the persistent Chroma volume; it then copied the output from the
temporary container into the smoke directory:

```bash
docker run --name rag-chatbot-gpu-smoke --network rag-chatbot_default \
  --volume "$PWD:/workspace:ro" --volume rag-chatbot_benchmark-data:/app/.benchmark-data \
  --workdir /workspace --env PYTHONPATH=/workspace/src:/workspace \
  --env GPU_SMOKE_OUTPUT=/tmp/gpu-smoke-results.json --env OLLAMA_BASE_URL=http://ollama:11434 \
  --env DOCUMENTS_DIR=/workspace/knowledge-base --env DATA_DIR=/app/.benchmark-data \
  --env CHROMA_DB_DIR=/app/.benchmark-data/chroma --env VECTOR_STORE_PROVIDER=chroma \
  --env LLM_PROVIDER=ollama --env EMBEDDING_PROVIDER=ollama \
  --env GENERATION_MODEL=llama3.2:3b --env EMBEDDING_MODEL=nomic-embed-text \
  --env GENERATION_TEMPERATURE=0 --env RERANKER_DEVICE=cpu \
  --env MAX_CONCURRENT_INFERENCES=1 --env OLLAMA_NUM_PARALLEL=1 \
  --env OLLAMA_MAX_LOADED_MODELS=1 --env AUTO_INDEX_ON_STARTUP=false \
  --entrypoint python rag-chatbot-benchmark:local evaluation/smoke/gpu/run_smoke.py
docker cp rag-chatbot-gpu-smoke:/tmp/gpu-smoke-results.json evaluation/smoke/gpu/results.json
docker rm rag-chatbot-gpu-smoke
```

For future use, rebuild the benchmark image after code changes and run only this script through Compose.
The GPU override bind-mounts `evaluation/smoke/gpu/` for its output; set the container UID/GID to the
host user if the directory is not writable by UID 10001. The Compose build/run form below is documented
for repeat use but was not executed verbatim in this smoke; the direct Docker invocation above was.

```bash
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full build benchmark-full
BENCHMARK_UID=$(id -u) BENCHMARK_GID=$(id -g) docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full run --rm --no-deps --entrypoint python benchmark-full evaluation/smoke/gpu/run_smoke.py
```

The tested engine smoke used the same image and persistent index, mounting the current source tree so
uncommitted fixes were active. It answered `datapizza-medium-014` in 4.976 seconds with citations aligned
to returned sources. A separate registered search passed `top_k="10"` and returned six documents. A
second development example (`datapizza-easy-016`) was refused; this smoke is not an accuracy evaluation.
The records and hardware observations are in `evaluation/smoke/gpu/` and are not official benchmark data.

Stop the benchmark service without deleting downloaded models or the index. Omitting the GPU override is
the explicit CPU-compatible configuration:

```bash
docker compose -f docker-compose.benchmark.yml -f docker-compose.benchmark.gpu.yml --profile full stop ollama
docker compose -f docker-compose.benchmark.yml --profile full config --quiet
```

Do not use `down -v` for normal shutdown; it deletes persistent volumes. Before any future full benchmark,
re-run the dataset validation and preflight, confirm `ollama ps` GPU placement with one small request,
check free VRAM, and execute scenarios sequentially. Keep `MAX_CONCURRENT_INFERENCES=1`,
`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, and `RERANKER_DEVICE=cpu` for this 4 GB card.
Compose defaults local inference timeouts to `REPLY_TIMEOUT_SECONDS=180`,
`OLLAMA_CONNECT_TIMEOUT_SECONDS=10`, and `OLLAMA_READ_TIMEOUT_SECONDS=180`.

The benchmark image mounts the corpus read-only and persists index data and model storage in named volumes.
Changing the corpus or chunking settings causes the application's index fingerprint checks to rebuild an
incompatible index rather than silently reusing it. Remove only the benchmark volumes when intentionally
resetting local state:

```bash
docker compose -f docker-compose.benchmark.yml --profile bm25 down -v
```

The Docker image is non-root. If a bind-mounted result directory is not writable by the container user,
set `BENCHMARK_UID=$(id -u)` and `BENCHMARK_GID=$(id -g)` before the Compose command, or pre-create the
directory with ownership for UID/GID 10001. Do not run the benchmark as a privileged container.

## Pilot protocol

For a new experiment, validate the dataset, run static preflight and minimal provider smoke tests, run the
development split, freeze its effective configuration, and only then run the held-out split. The checked-in
dataset and annotations are immutable. Do not tune after looking at held-out answers. The historical and
current answer-quality datasets are distinct; never attach historical judge scores to new generations.

To export the current selected runs into a **new, unused** versioned destination (the exporter replaces
an existing destination directory):

```bash
python3 -m evaluation.export_analysis_bundle \
  --results evaluation/results/datapizza-current \
  --dataset evaluation/dataset/datapizza \
  --corpus knowledge-base \
  --selection evaluation/published/datapizza-current.selection.json \
  --published evaluation/published/datapizza-current-rerun-<id> \
  --no-zip --output /tmp/datapizza-current.zip
```

The result comparison is at `evaluation/comparisons/baseline-vs-current/REPORT.md`. Current latency is
CPU-only and should not be compared directly with the historical GPU-enabled run. The current judge review
template is intentionally blank until a consistent reviewer is available.
