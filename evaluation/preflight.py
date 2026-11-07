"""Read-only checks to run before a live Datapizza benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .adapters.datapizza import validate_dataset


def run_preflight(
    *,
    configuration: str,
    mode: str,
    corpus: Path,
    dataset: Path,
    output: Path,
) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    try:
        errors = validate_dataset(corpus, dataset)
        if errors:
            checks.append(("dataset", "FAIL", "; ".join(errors)))
        else:
            checks.append(("dataset", "PASS", "20 documents and 56 questions are valid"))
    except (OSError, ValueError, KeyError) as exc:
        checks.append(("dataset", "FAIL", str(exc)))
    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output, prefix=".preflight-", delete=True):
            pass
        checks.append(("output", "PASS", str(output)))
    except OSError as exc:
        checks.append(("output", "FAIL", f"Output directory is not writable: {exc}"))
    valid_configurations = {"standalone", "bm25", "dense", "hybrid", "hybrid-rerank", "full"}
    valid_modes = {"retrieval-only", "end-to-end"}
    if configuration not in valid_configurations:
        checks.append(("configuration", "FAIL", f"Unsupported configuration: {configuration}"))
    elif configuration in {"standalone", "full"} and mode != "end-to-end":
        checks.append(("mode", "FAIL", f"{configuration} only supports end-to-end"))
    elif mode not in valid_modes:
        checks.append(("mode", "FAIL", f"Unsupported mode: {mode}"))
    else:
        checks.append(("configuration", "PASS", f"{configuration}/{mode}"))

    if configuration in {"dense", "hybrid", "hybrid-rerank", "full"}:
        checks.append(
            ("chroma", "PASS" if importlib.util.find_spec("chromadb") else "UNAVAILABLE", "Static import check only")
        )
    if configuration == "hybrid-rerank":
        checks.append(
            (
                "reranker",
                "PASS" if importlib.util.find_spec("sentence_transformers") else "UNAVAILABLE",
                "Model weights not loaded",
            )
        )
    if mode == "end-to-end":
        generation_model = os.getenv("GENERATION_MODEL", "llama3.2:3b").strip()
        checks.append(
            (
                "generation_model",
                "NOT CHECKED" if generation_model else "FAIL",
                generation_model or "Set GENERATION_MODEL before running an end-to-end experiment",
            )
        )
    if configuration not in {"standalone", "bm25"}:
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text").strip()
        checks.append(
            (
                "embedding_model",
                "NOT CHECKED" if embedding_model else "FAIL",
                embedding_model or "Set EMBEDDING_MODEL before running retrieval",
            )
        )
    if shutil.which("docker"):
        compose = subprocess.run(
            ["docker", "compose", "-f", "docker-compose.benchmark.yml", "config", "--quiet"],
            check=False,
            capture_output=True,
            text=True,
        )
        checks.append(
            (
                "docker_compose",
                "PASS" if compose.returncode == 0 else "FAIL",
                "Static Compose validation only",
            )
        )
    else:
        checks.append(("docker_compose", "UNAVAILABLE", "Docker is not installed; run docker compose config manually"))
    if os.getenv("OLLAMA_BASE_URL"):
        checks.append(("ollama", "NOT CHECKED", "Connectivity is not probed by static preflight"))
    else:
        checks.append(("ollama", "NOT CHECKED", "Set OLLAMA_BASE_URL for an explicit runtime check"))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only benchmark preflight checks.")
    parser.add_argument(
        "--configuration", choices=("standalone", "bm25", "dense", "hybrid", "hybrid-rerank", "full"), default="bm25"
    )
    parser.add_argument("--mode", choices=("retrieval-only", "end-to-end"), default="retrieval-only")
    parser.add_argument("--corpus", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/datapizza"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/datapizza"))
    args = parser.parse_args()
    checks = run_preflight(**vars(args))
    for name, status, message in checks:
        print(f"{status}: {name}: {message}")
    return 1 if any(status == "FAIL" for _name, status, _message in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
