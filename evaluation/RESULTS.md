# Evaluation Results

## Measured baseline

The only reproducible experiment completed in the available environment is the deterministic BM25 retrieval baseline on the `dev` split.

| Configuration | Split | Questions | Recall@5 | MRR@5 | nDCG@5 | Complete evidence | Retrieval p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | dev | 10 | 1.000 | 0.889 | 0.918 | 1.000 | 0.12 ms |

All 10 questions completed successfully. The measurement used the version-controlled six-document corpus, 800-character chunks with 80-character overlap, and no model inference. Latency is process-local retrieval time and excludes indexing, embedding, and generation.

## Reproduction

```bash
python evaluation/validate_dataset.py
python -m evaluation.benchmark \
  --configuration bm25 \
  --split dev \
  --top-k 5 \
  --output /tmp/rag-bm25-dev.json
```

Live standalone, dense, hybrid, reranking, and full-pipeline runs are available through the explicit `--live` CLI path when the required provider, vector store, and model resources are configured. They were not executed for this report because those resources were unavailable. Consequently, this report makes no comparative quality, groundedness, or optimization claim.

## Validation status

The deterministic validation suite passed on 2026-11-07: 34 tests, 42.51% application coverage, dataset validation, Ruff, package build, and the BM25 smoke benchmark. Live model comparisons remain pending because the local Ollama executable could not connect to `127.0.0.1:11434`, and the optional Chroma, Ollama provider, and reranker packages were not installed in the validation environment.

No live-model results or optimization claims are reported until those resources are available.
