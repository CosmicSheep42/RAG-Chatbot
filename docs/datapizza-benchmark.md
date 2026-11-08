# Datapizza Benchmark and Reproduction Guide

The Datapizza benchmark has been executed locally for all supported configurations. This document now
serves as the reproduction guide for the completed measurements, not only as preparation instructions.
The published per-question benchmark records are in `evaluation/published/datapizza/`; the separate
single-pass LLM-as-a-Judge results are in `evaluation/judge/datapizza/`. No new inference is required to
inspect or verify either set of records.

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

Run one scenario at a time. These commands start containers and reproduce new measurements; they are not
needed to inspect the already-published results:

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

For NVIDIA Container Toolkit, add `-f docker-compose.benchmark.gpu.yml`; otherwise the base file is
CPU-compatible. Set `RERANKER_DEVICE=cpu` (the default) to keep the cross-encoder off the 4 GB GPU.
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

When live execution is intentionally started later: run static preflight, run one or two-question
provider smoke tests per scenario, run the 12-question development split, freeze the effective manifest,
run the 44-question test split, then export and review answers with the manual rubric. Do not tune after
looking at held-out test results.
