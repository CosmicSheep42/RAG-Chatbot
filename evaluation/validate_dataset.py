"""Validate the fixed evaluation corpus and its JSONL ground truth."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CATEGORIES = {
    "direct_factual_lookup",
    "multi_document_reasoning",
    "paraphrased_query",
    "unanswerable",
    "conflicting_version",
    "long_context",
}
SPLITS = {"dev", "test"}
REQUIRED_FIELDS = {
    "id",
    "question",
    "category",
    "split",
    "answerable",
    "reference_answer",
    "relevant_documents",
    "evidence",
    "required_facts",
}


def _sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return {heading: "\n".join(lines) for heading, lines in sections.items()}


def _error(index: int, message: str) -> str:
    return f"example {index}: {message}"


def validate_dataset(corpus_dir: Path, dataset_path: Path) -> list[str]:
    errors: list[str] = []
    documents = {path.relative_to(corpus_dir).as_posix(): path for path in sorted(corpus_dir.rglob("*.md"))}
    corpus_sections = {source: _sections(path.read_text(encoding="utf-8")) for source, path in documents.items()}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()

    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [f"cannot read dataset: {exc}"]

    for index, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(_error(index, "blank lines are not valid JSONL entries"))
            continue
        try:
            example: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(_error(index, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(example, dict):
            errors.append(_error(index, "entry must be a JSON object"))
            continue

        missing = REQUIRED_FIELDS - example.keys()
        if missing:
            errors.append(_error(index, f"missing fields: {', '.join(sorted(missing))}"))
            continue

        example_id = example["id"]
        question = example["question"]
        if not isinstance(example_id, str) or not example_id.strip():
            errors.append(_error(index, "id must be a nonempty string"))
        elif example_id in seen_ids:
            errors.append(_error(index, f"duplicate id: {example_id}"))
        else:
            seen_ids.add(example_id)

        if not isinstance(question, str) or not question.strip():
            errors.append(_error(index, "question must be a nonempty string"))
        else:
            normalized_question = " ".join(question.casefold().split())
            if normalized_question in seen_questions:
                errors.append(_error(index, "duplicate question"))
            seen_questions.add(normalized_question)

        if example.get("category") not in CATEGORIES:
            errors.append(_error(index, f"invalid category: {example.get('category')}"))
        if example.get("split") not in SPLITS:
            errors.append(_error(index, f"invalid split: {example.get('split')}"))
        if not isinstance(example.get("answerable"), bool):
            errors.append(_error(index, "answerable must be a boolean"))

        answerable = example.get("answerable") is True
        reference_answer = example.get("reference_answer")
        required_facts = example.get("required_facts")
        evidence = example.get("evidence")
        relevant_documents = example.get("relevant_documents")
        if answerable and (not isinstance(reference_answer, str) or not reference_answer.strip()):
            errors.append(_error(index, "answerable examples need a reference_answer"))
        if not answerable and reference_answer is not None:
            errors.append(_error(index, "unanswerable examples must have a null reference_answer"))
        if not isinstance(required_facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in required_facts
        ):
            errors.append(_error(index, "required_facts must be a list of nonempty strings"))
        elif answerable and not required_facts:
            errors.append(_error(index, "answerable examples need required_facts"))
        elif not answerable and required_facts:
            errors.append(_error(index, "unanswerable examples cannot have required_facts"))
        if not isinstance(relevant_documents, list) or any(not isinstance(doc, str) for doc in relevant_documents):
            errors.append(_error(index, "relevant_documents must be a list of strings"))
        relevant_document_list = relevant_documents if isinstance(relevant_documents, list) else []
        if not isinstance(evidence, list):
            errors.append(_error(index, "evidence must be a list"))
            evidence = []
        if not answerable and (relevant_documents or evidence):
            errors.append(_error(index, "unanswerable examples cannot cite corpus evidence"))
        if answerable and not evidence:
            errors.append(_error(index, "answerable examples need evidence"))

        for document in relevant_document_list:
            if document not in documents:
                errors.append(_error(index, f"missing source document: {document}"))
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"document", "section", "text"}:
                errors.append(_error(index, "evidence must contain only document, section, and text"))
                continue
            source = item["document"]
            section = item["section"]
            text = item["text"]
            if source not in documents:
                errors.append(_error(index, f"evidence references missing document: {source}"))
                continue
            if not isinstance(section, str) or section not in corpus_sections[source]:
                errors.append(_error(index, f"evidence references missing section: {source}#{section}"))
                continue
            if not isinstance(text, str) or not text.strip() or text not in corpus_sections[source][section]:
                errors.append(_error(index, f"evidence text is not present in {source}#{section}"))
            if source not in relevant_document_list:
                errors.append(_error(index, f"evidence document is not listed as relevant: {source}"))

    if not lines:
        errors.append("dataset is empty")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase 2 evaluation dataset.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset" / "questions.jsonl")
    args = parser.parse_args()
    errors = validate_dataset(args.corpus, args.dataset)
    if errors:
        print("Evaluation dataset validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Evaluation dataset is valid: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
