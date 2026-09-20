from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .models import BenchmarkCase, TurnInput


ORACLE_FIELDS = {"answer", "answer_session_ids", "has_answer"}

QUESTION_TYPE_ALIASES = {
    "user": "single-session-user",
    "single-session-user": "single-session-user",
    "assistant": "single-session-assistant",
    "single-session-assistant": "single-session-assistant",
    "preference": "single-session-preference",
    "single-session-preference": "single-session-preference",
    "temporal": "temporal-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "multi": "multi-session",
    "multi-session": "multi-session",
    "knowledge": "knowledge-update",
    "update": "knowledge-update",
    "knowledge-update": "knowledge-update",
}


def normalize_question_type(value: str) -> str:
    cleaned = value.strip().lower()
    return QUESTION_TYPE_ALIASES.get(cleaned, cleaned)


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else list(value.get("data", value.get("cases", [])))


def load_cases(
    path: Path,
    case_id: str | None = None,
    limit: int | None = None,
    question_type: str | Iterable[str] | None = None,
) -> Iterator[BenchmarkCase]:
    count = 0
    allowed_types = None
    if question_type:
        if isinstance(question_type, str):
            types_list = [t.strip() for t in question_type.split(",") if t.strip()]
        else:
            types_list = list(question_type)
        allowed_types = {normalize_question_type(t) for t in types_list}

    for raw in _records(path):
        qid = str(raw.get("question_id", raw.get("id", "")))
        if case_id and qid != case_id:
            continue
        raw_qtype = str(raw.get("question_type", ""))
        if allowed_types is not None and normalize_question_type(raw_qtype) not in allowed_types:
            continue
        sessions = raw.get("haystack_sessions", [])
        session_ids = raw.get("haystack_session_ids", [str(i) for i in range(len(sessions))])
        dates = raw.get("haystack_dates", [None] * len(sessions))
        turns: list[TurnInput] = []
        answer_turns: list[tuple[str, int]] = []
        for si, session in enumerate(sessions):
            sid = str(session_ids[si])
            for ti, turn in enumerate(session):
                content = turn.get("content")
                role = turn.get("role")
                if not isinstance(content, str) or not isinstance(role, str):
                    raise ValueError(f"Invalid turn metadata in {qid}/{sid}/{ti}")
                if turn.get("has_answer") is True:
                    answer_turns.append((sid, ti))
                turns.append(TurnInput(sid, si, ti, role, content, dates[si] if si < len(dates) else None))
        yield BenchmarkCase(
            case_id=qid, question_type=str(raw.get("question_type", "")), question=str(raw["question"]),
            question_date=raw.get("question_date"), turns=tuple(turns), reference_answer=raw.get("answer"),
            answer_session_ids=tuple(str(x) for x in raw.get("answer_session_ids", [])),
            answer_turn_keys=tuple(answer_turns),
        )
        count += 1
        if limit is not None and count >= limit:
            return

