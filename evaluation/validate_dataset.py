"""Validate the pinned Datapizza D&D 5e evaluation data offline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .adapters.datapizza import validate_dataset
except ImportError:  # Support the documented `python evaluation/validate_dataset.py` command.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.adapters.datapizza import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Datapizza evaluation dataset.")
    parser.add_argument("--corpus", type=Path, default=Path("knowledge-base"))
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/dataset/datapizza"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    errors = validate_dataset(args.corpus, args.dataset, args.manifest)
    if errors:
        print("Datapizza evaluation dataset validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Datapizza evaluation dataset is valid: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
