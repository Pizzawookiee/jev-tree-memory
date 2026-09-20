from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .judge_prompt import build_judge_prompt


def prediction_hash(hypothesis: str) -> str:
    return hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()


def _existing_entries(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            entries[item["manual_judge_id"]] = item
    return entries


def build_manual_entry(case, artifact: dict, model: str) -> dict:
    hypothesis = artifact.get("answer")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError(f"No generated answer for {case.case_id}:{artifact['mode']}")
    prompt = build_judge_prompt(case.case_id, case.question, str(case.reference_answer), hypothesis,
                                question_type=case.question_type)
    return {
        "manual_judge_id": f"{case.case_id}:{artifact['mode']}",
        "question_id": case.case_id,
        "mode": artifact["mode"],
        "model_requested": model,
        "prompt_version": prompt.prompt_version,
        "system_prompt": prompt.system_prompt,
        "user_prompt": prompt.user_prompt,
        "expected_response_format": prompt.expected_response_format,
        "prediction_hash": prediction_hash(hypothesis),
        "manual_response": None,
        "manual_label": None,
    }


def export_manual_prompts(records: list[tuple[object, dict]], jsonl_path: Path, model: str) -> tuple[Path, Path, int]:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    prior = _existing_entries(jsonl_path)
    # Preserve entries from earlier partial exports; regenerated IDs replace
    # them in place, so resume never duplicates or discards completed labels.
    entries: dict[str, dict] = dict(prior)
    for case, artifact in records:
        entry = build_manual_entry(case, artifact, model)
        old = prior.get(entry["manual_judge_id"])
        if old and old.get("prediction_hash") == entry["prediction_hash"]:
            entry["manual_response"] = old.get("manual_response")
            entry["manual_label"] = old.get("manual_label")
        entries[entry["manual_judge_id"]] = entry
    ordered = [entries[key] for key in sorted(entries)]
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
    temporary.replace(jsonl_path)
    markdown_path = jsonl_path.with_suffix(".md")
    sections = []
    for index, entry in enumerate(ordered, 1):
        exact = entry["user_prompt"]
        if entry["system_prompt"]:
            exact = f"System:\n{entry['system_prompt']}\n\nUser:\n{exact}"
        sections.append(
            f"# Manual Judge Prompt {index}\n\nManual judge ID: {entry['manual_judge_id']}\n\n"
            f"## Copy everything below\n\n{exact}\n\n## Expected response\n\n"
            f"{entry['expected_response_format']}\n"
        )
    markdown_path.write_text("\n---\n\n".join(sections), encoding="utf-8")
    return jsonl_path, markdown_path, len(ordered)
