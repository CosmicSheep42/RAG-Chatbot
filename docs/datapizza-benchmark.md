# Datapizza Benchmark Preparation

The repository contains 20 Datapizza D&D 5e Markdown documents in `knowledge-base/` and 56 original QA
examples in `evaluation/dataset/datapizza/`. The split manifest contains 12 development and 44 test IDs.
The benchmark adapter preserves the upstream passage text and character offsets. One upstream medium
example has no passages; it remains preserved and passage metrics are not applicable for that example.

Validate the copied data without starting any service:

```bash
python evaluation/validate_dataset.py
```

Build the benchmark image only when preparing a machine for a future run. The build does not download an
LLM or execute a benchmark. Add `--build-arg INSTALL_RERANKER=true` when preparing the reranking scenario.

The default profile is sequential and uses `llama3.2:3b`, `nomic-embed-text`, Chroma, temperature `0`, and
one concurrent inference. Models must be prepared separately in the persistent Ollama volume:

```bash
docker compose -f docker-compose.benchmark.yml --profile standalone build
docker compose -f docker-compose.benchmark.yml --profile standalone up -d ollama
docker compose -f docker-compose.benchmark.yml --profile standalone exec ollama ollama pull llama3.2:3b
docker compose -f docker-compose.benchmark.yml --profile standalone exec ollama ollama pull nomic-embed-text
```

Run one scenario at a time. These commands are intentionally not run by this migration:

```bash
docker compose -f docker-compose.benchmark.yml --profile standalone run --rm benchmark-standalone
docker compose -f docker-compose.benchmark.yml --profile bm25 run --rm benchmark-bm25
docker compose -f docker-compose.benchmark.yml --profile dense run --rm benchmark-dense
docker compose -f docker-compose.benchmark.yml --profile hybrid run --rm benchmark-hybrid
docker compose -f docker-compose.benchmark.yml --profile hybrid-rerank run --rm benchmark-hybrid-rerank
docker compose -f docker-compose.benchmark.yml --profile full run --rm benchmark-full
```

Use `BENCHMARK_SPLIT=test` for the held-out split. The result files are written below
`evaluation/results/datapizza/`, which is ignored by Git. Retrieval-only BM25 needs no Ollama service;
the other profiles use Ollama and require the corresponding local models. For NVIDIA Container Toolkit,
add `-f docker-compose.benchmark.gpu.yml`; otherwise the base file is CPU-compatible. The reranker runs
on the application CPU when installed.

The benchmark image mounts the corpus read-only and persists index data and model storage in named volumes.
Changing the corpus or chunking settings causes the application's index fingerprint checks to rebuild an
incompatible index rather than silently reusing it. Remove only the benchmark volumes when intentionally
resetting local state:

```bash
docker compose -f docker-compose.benchmark.yml --profile bm25 down -v
```
