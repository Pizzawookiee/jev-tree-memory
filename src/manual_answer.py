from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .answer import ANSWER_INSTRUCTIONS


def retrieval_hash(question: str, evidence: str) -> str:
    payload = f"{question}\n\0{evidence}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _existing_entries(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            entries[item["manual_answer_id"]] = item
    return entries


def build_manual_answer_entry(case, artifact: dict, model: str) -> dict:
    evidence = artifact.get("packed_context")
    if not isinstance(evidence, str):
        raise ValueError(f"No packed retrieval context for {case.case_id}:{artifact['mode']}")
    return {
        "manual_answer_id": f"{case.case_id}:{artifact['mode']}",
        "question_id": case.case_id,
        "mode": artifact["mode"],
        "model_requested": model,
        "system_prompt": ANSWER_INSTRUCTIONS,
        "user_prompt": f"Question:\n{case.question}\n\nRetrieved evidence:\n{evidence}",
        "retrieval_hash": retrieval_hash(case.question, evidence),
        "suggested_answer": None,
    }


def export_manual_answer_prompts(
    records: list[tuple[object, dict]], jsonl_path: Path, model: str
) -> tuple[Path, Path, int]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _existing_entries(jsonl_path)
    entries: dict[str, dict] = dict(prior)
    for case, artifact in records:
        entry = build_manual_answer_entry(case, artifact, model)
        old = prior.get(entry["manual_answer_id"])
        if old and old.get("retrieval_hash") == entry["retrieval_hash"]:
            entry["suggested_answer"] = old.get("suggested_answer")
        entries[entry["manual_answer_id"]] = entry

    ordered = [entries[key] for key in sorted(entries)]
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
        encoding="utf-8",
    )
    temporary.replace(jsonl_path)

    markdown_path = jsonl_path.with_suffix(".md")
    introduction = (
        "# Manual answer generation\n\n"
        "For every prompt below, answer using only its retrieved evidence. Put the result in the "
        "matching JSONL entry's `suggested_answer` field; do not change IDs, prompts, or hashes.\n"
    )
    sections = []
    for index, entry in enumerate(ordered, 1):
        sections.append(
            f"# Answer Prompt {index}\n\nManual answer ID: {entry['manual_answer_id']}\n\n"
            f"## System\n\n{entry['system_prompt']}\n\n"
            f"## User\n\n{entry['user_prompt']}\n"
        )
    markdown_path.write_text(introduction + "\n---\n\n".join(sections), encoding="utf-8")
    return jsonl_path, markdown_path, len(ordered)


def load_manual_answers(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise ValueError(f"Manual answer input does not exist: {path}")
    answers: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        answer_id = entry.get("manual_answer_id")
        if not isinstance(answer_id, str) or not answer_id:
            raise ValueError(f"Missing manual_answer_id on line {line_number} of {path}")
        if answer_id in answers:
            raise ValueError(f"Duplicate manual_answer_id in {path}: {answer_id}")
        answers[answer_id] = entry
    return answers
