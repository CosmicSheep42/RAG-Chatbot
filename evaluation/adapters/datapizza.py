"""Offline loader and validator for the Datapizza D&D 5e evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_DOCUMENT_COUNT = 20
EXPECTED_ENTRY_COUNT = 56
EXPECTED_SOURCE_REVISION = "f2428dce66ea418a785998fa6ec57bc561b72239"
# The upstream medium question with ID 4 intentionally has no annotated passages.
KNOWN_MISSING_PASSAGES = {"medium": {4}}


@dataclass(frozen=True)
class Passage:
    content: str
    document_path: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Example:
    id: str
    source_id: int
    tier: str
    question: str
    answer: str
    passages: tuple[Passage, ...]
    split: str

    @property
    def relevant_documents(self) -> list[str]:
        return list(dict.fromkeys(passage.document_path for passage in self.passages))


def load_examples(dataset_dir: Path, split: str, manifest_path: Path | None = None) -> list[Example]:
    manifest = _load_manifest(manifest_path or dataset_dir / "splits.json")
    selected = set(manifest[split])
    examples: list[Example] = []
    for tier in ("easy", "medium"):
        entries = json.loads((dataset_dir / f"{tier}.json").read_text(encoding="utf-8"))
        for item in entries:
            example_id = f"datapizza-{tier}-{int(item['id']):03d}"
            if example_id not in selected:
                continue
            examples.append(
                Example(
                    id=example_id,
                    source_id=int(item["id"]),
                    tier=tier,
                    question=item["question"],
                    answer=item["answer"],
                    passages=tuple(Passage(**passage) for passage in item["passages"]),
                    split=split,
                )
            )
    return sorted(examples, key=lambda example: example.id)


def validate_dataset(corpus_dir: Path, dataset_dir: Path, manifest_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    documents = sorted(corpus_dir.glob("*.md"))
    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        errors.append(f"expected {EXPECTED_DOCUMENT_COUNT} Markdown documents, found {len(documents)}")
    corpus = {path.name: path.read_text(encoding="utf-8") for path in documents}
    all_ids: set[str] = set()
    questions: set[str] = set()
    all_examples: dict[str, dict[str, Any]] = {}
    for tier in ("easy", "medium"):
        path = dataset_dir / f"{tier}.json"
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: cannot read JSON: {exc}")
            continue
        if not isinstance(entries, list):
            errors.append(f"{path}: top-level value must be a list")
            continue
        for index, item in enumerate(entries):
            prefix = f"{tier}[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix}: entry must be an object")
                continue
            example_id = f"datapizza-{tier}-{item.get('id', ''):03d}" if isinstance(item.get("id"), int) else ""
            if not example_id:
                errors.append(f"{prefix}: id must be an integer")
            elif example_id in all_ids:
                errors.append(f"{prefix}: duplicate normalized id {example_id}")
            else:
                all_ids.add(example_id)
                all_examples[example_id] = item
            if not isinstance(item.get("question"), str) or not item["question"].strip():
                errors.append(f"{prefix}: question must be nonempty")
            else:
                normalized_question = " ".join(item["question"].casefold().split())
                if normalized_question in questions:
                    errors.append(f"{prefix}: duplicate normalized question")
                questions.add(normalized_question)
            if not isinstance(item.get("answer"), str) or not item["answer"].strip():
                errors.append(f"{prefix}: answer must be nonempty")
            passages = item.get("passages")
            if not isinstance(passages, list) or (not passages and item.get("id") not in KNOWN_MISSING_PASSAGES[tier]):
                errors.append(f"{prefix}: passages must be a nonempty list")
                continue
            for passage_index, passage in enumerate(passages):
                passage_prefix = f"{prefix}.passages[{passage_index}]"
                if not isinstance(passage, dict) or set(passage) != {
                    "content",
                    "document_path",
                    "start_char",
                    "end_char",
                }:
                    errors.append(f"{passage_prefix}: invalid passage fields")
                    continue
                source = passage["document_path"]
                start = passage["start_char"]
                end = passage["end_char"]
                if source not in corpus:
                    errors.append(f"{passage_prefix}: missing source document {source}")
                    continue
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(corpus[source])
                ):
                    errors.append(f"{passage_prefix}: invalid character offsets")
                elif corpus[source][start:end] != passage["content"]:
                    errors.append(f"{passage_prefix}: content does not match character offsets")
    if len(all_ids) != EXPECTED_ENTRY_COUNT:
        errors.append(f"expected {EXPECTED_ENTRY_COUNT} QA entries, found {len(all_ids)}")
    errors.extend(_validate_manifest(manifest_path or dataset_dir / "splits.json", all_ids, all_examples))
    return errors


def _load_manifest(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"dev": list(data["dev"]), "test": list(data["test"])}


def _validate_manifest(path: Path, all_ids: set[str], examples: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot read split manifest: {exc}"]
    if manifest.get("source_revision") != EXPECTED_SOURCE_REVISION:
        errors.append(f"{path}: unexpected source revision")
    dev = manifest.get("dev")
    test = manifest.get("test")
    if not isinstance(dev, list) or not isinstance(test, list):
        return errors + [f"{path}: dev and test must be lists"]
    if len(dev) != 12 or len(test) != 44:
        errors.append(f"{path}: expected 12 dev and 44 test IDs")
    if set(dev) & set(test):
        errors.append(f"{path}: dev and test overlap")
    if set(dev) | set(test) != all_ids:
        errors.append(f"{path}: split IDs do not cover the complete dataset")
    if any(identifier not in examples for identifier in dev + test):
        errors.append(f"{path}: split contains an unknown ID")
    return errors
