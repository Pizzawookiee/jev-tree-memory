from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import requests

from .answer import ModelResult
from .judge_prompt import build_judge_prompt


OFFICIAL_JUDGE_ALIAS = "gpt-4o"
OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array or JSONL records in {path}")
    return value


def export_hypotheses(records: Iterable[tuple[object, dict]], path: Path, mode: str) -> tuple[Path, int]:
    """Write the exact two-field LongMemEval hypothesis format."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for case, artifact in records:
        if artifact.get("mode") != mode:
            continue
        question_id = str(case.case_id)
        hypothesis = artifact.get("answer")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"No answer is available for {question_id}/{mode}")
        if question_id in seen:
            raise ValueError(f"Duplicate LongMemEval question_id in export: {question_id}")
        seen.add(question_id)
        entries.append({"question_id": question_id, "hypothesis": hypothesis.strip()})
    if not entries:
        raise ValueError(f"No {mode} answers were available for LongMemEval export")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path, len(entries)


class OfficialLongMemEvalEvaluator:
    """Native equivalent of LongMemEval's official evaluate_qa.py GPT-4o path."""

    def __init__(self, api_key: str, session: requests.Session | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for official LongMemEval evaluation")
        self.api_key = api_key
        self.session = session or requests.Session()

    def _judge(self, reference: dict, hypothesis: str) -> tuple[bool, ModelResult]:
        prompt = build_judge_prompt(
            str(reference["question_id"]),
            str(reference["question"]),
            str(reference["answer"]),
            hypothesis,
            question_type=str(reference["question_type"]),
        )
        payload = {
            "model": OFFICIAL_JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt.user_prompt}],
            "n": 1,
            "temperature": 0,
            "max_tokens": 10,
        }
        started = time.perf_counter()
        response = None
        for attempt in range(6):
            response = self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            if response.status_code != 429 and response.status_code < 500:
                break
            if attempt < 5:
                time.sleep(min(2**attempt, 30))
        assert response is not None
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"].strip()
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        estimated_cost = input_tokens / 1_000_000 * 2.50 + output_tokens / 1_000_000 * 10.00
        return "yes" in text.lower(), ModelResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            estimated_cost=estimated_cost,
        )

    def evaluate(self, hypothesis_path: Path, reference_path: Path) -> tuple[Path, Path, dict]:
        hypotheses = _read_records(hypothesis_path)
        references = _read_records(reference_path)
        reference_by_id = {str(item["question_id"]): item for item in references}
        logs: list[dict] = []
        scores: dict[str, list[int]] = defaultdict(list)
        abstention_scores: list[int] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0
        total_cost = 0.0

        result_path = Path(str(hypothesis_path) + f".eval-results-{OFFICIAL_JUDGE_ALIAS}")
        for index, entry in enumerate(hypotheses, start=1):
            if set(entry) != {"question_id", "hypothesis"}:
                raise ValueError(
                    f"Official hypothesis entry {index} must contain exactly question_id and hypothesis"
                )
            question_id = str(entry["question_id"])
            if question_id not in reference_by_id:
                raise ValueError(f"Question {question_id} is missing from reference file {reference_path}")
            reference = reference_by_id[question_id]
            correct, usage = self._judge(reference, str(entry["hypothesis"]))
            logged = dict(entry)
            logged["autoeval_label"] = {"model": OFFICIAL_JUDGE_MODEL, "label": correct}
            logs.append(logged)
            value = 1 if correct else 0
            scores[str(reference["question_type"])].append(value)
            if "_abs" in question_id:
                abstention_scores.append(value)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_latency_ms += usage.latency_ms
            total_cost += usage.estimated_cost
            print(
                f"Official LongMemEval judge: {index:,}/{len(hypotheses):,} "
                f"({index / len(hypotheses) * 100:.1f}%)",
                flush=True,
            )

        result_path.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in logs),
            encoding="utf-8",
        )
        all_values = [value for values in scores.values() for value in values]
        task_values = [sum(scores[k]) / len(scores[k]) for k in QUESTION_TYPES if scores.get(k)]
        by_type = {
            key: {"accuracy": sum(scores[key]) / len(scores[key]), "count": len(scores[key])}
            for key in QUESTION_TYPES
            if scores.get(key)
        }
        metrics = {
            "evaluator_alias": OFFICIAL_JUDGE_ALIAS,
            "evaluator_model": OFFICIAL_JUDGE_MODEL,
            "hypothesis_file": str(hypothesis_path),
            "reference_file": str(reference_path),
            "evaluated": len(logs),
            "task_averaged_accuracy": sum(task_values) / len(task_values) if task_values else None,
            "overall_accuracy": sum(all_values) / len(all_values) if all_values else None,
            "abstention_accuracy": (
                sum(abstention_scores) / len(abstention_scores) if abstention_scores else None
            ),
            "abstention_count": len(abstention_scores),
            "by_question_type": by_type,
            "judge_token_usage": {"input": total_input_tokens, "output": total_output_tokens},
            "judge_latency_ms": total_latency_ms,
            "judge_estimated_cost": total_cost,
        }
        metrics_path = Path(str(result_path) + ".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return result_path, metrics_path, metrics
