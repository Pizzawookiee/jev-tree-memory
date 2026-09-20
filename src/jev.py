from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from typing import Any

import requests

from .database import MemoryDB
from .jev_cache import JevDecisionCache
from .models import MemoryDecision
from .tree import ROOTS, MemoryTree

MEMORY_TYPES = {
    "fact": "A durable factual statement",
    "preference": "A like, dislike, habit, or choice",
    "event": "A time-bounded occurrence",
    "procedure": "Instructions or a repeatable method",
    "state": "A condition that may change",
    "other": "None of the other memory types",
}
PRIORITIES = {"low": "Unlikely to matter later", "normal": "Ordinary memory value", "high": "Likely important for future questions"}
TEMPORAL = {"timeless": "Not tied to a time", "historical": "True in the past", "current": "Describes the current state", "unknown": "Time scope is unclear"}
JEV_INPUT_USD_PER_MILLION_TOKENS = 0.042
TRIVIAL_ACKNOWLEDGEMENTS = {
    "acknowledged", "alright", "cool", "got it", "great", "makes sense", "noted",
    "okay", "ok", "sounds good", "sure", "thanks", "thank you", "understood",
    "will do", "youre welcome", "you're welcome",
}


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _topic_label(text: str) -> str:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)]
    stop = {"the", "and", "that", "this", "with", "from", "have", "your", "about", "will", "was", "are"}
    selected = [w for w in words if w not in stop][:3]
    return "-".join(selected) or "misc"


def _normalized_acknowledgement(text: str) -> str:
    return re.sub(r"[^a-z']+", " ", text.lower()).strip()


def _is_trivial_acknowledgement(role: str, text: str) -> bool:
    return role == "assistant" and _normalized_acknowledgement(text) in TRIVIAL_ACKNOWLEDGEMENTS


def _lean_memory_type(primary: str) -> str:
    return {
        "preferences": "preference", "procedures": "procedure", "episodes": "event",
        "projects": "state", "environment": "state", "user": "fact",
    }.get(primary, "other")


def _lean_temporal_scope(content: str, memory_type: str) -> str:
    lower = content.lower()
    if re.search(r"\b(currently|right now|now|latest|today)\b", lower):
        return "current"
    if re.search(r"\b(yesterday|last (?:week|month|year)|ago|previously|formerly)\b|\b\d{4}-\d{2}-\d{2}\b", lower):
        return "historical"
    if memory_type in {"fact", "preference", "procedure"}:
        return "timeless"
    return "unknown"


class JevClient:
    def __init__(self, api_key: str | None, model: str = "jev-latest", allow_fallback: bool = False,
                 session: requests.Session | None = None):
        # The explicit smoke-test flag forces a deterministic backend even when
        # the developer has real credentials in .env, preventing paid/network
        # calls during tests.
        self.api_key = None if allow_fallback else api_key
        self.model = model
        self.allow_fallback = allow_fallback
        self.session = session or requests.Session()
        self.backend = model if self.api_key else "deterministic-jev-fallback"
        self.request_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_latency_ms = 0.0
        if not self.api_key and not allow_fallback:
            raise RuntimeError("TYPESAFE_API_KEY is required for jev-primary")

    @staticmethod
    def cost_for_usage(usage: dict[str, Any]) -> float:
        return int(usage.get("input_tokens", 0) or 0) / 1_000_000 * JEV_INPUT_USD_PER_MILLION_TOKENS

    @property
    def estimated_cost(self) -> float:
        return self.input_tokens / 1_000_000 * JEV_INPUT_USD_PER_MILLION_TOKENS

    def ask(self, state: str, questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], float]:
        start = time.perf_counter()
        if not self.api_key:
            answers, usage = self._fallback(state, questions), {"input_tokens": 0, "output_tokens": 0}
        else:
            response = self.session.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"state": state, "model": self.model, "questions": questions}, timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            answers, usage = payload["answers"], payload.get("usage", {})
        latency = (time.perf_counter() - start) * 1000
        self.request_count += 1
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_latency_ms += latency
        return answers, usage, latency

    def ask_batch(
        self, items: list[tuple[str, dict[str, dict[str, Any]]]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        """Evaluate independent memory-unit routing question sets in one API request."""
        if not items:
            return [], {"input_tokens": 0, "output_tokens": 0}, 0.0
        if len(items) == 1:
            answers, usage, latency = self.ask(*items[0])
            return [answers], usage, latency

        flattened: dict[str, dict[str, Any]] = {}
        for index, (_memory_unit, questions) in enumerate(items):
            for name, question in questions.items():
                batched_question = dict(question)
                batched_question["instructions"] = {
                    "decision": question["instructions"],
                    "target": f"`memory_units[{index}].text`",
                    "constraint": "Evaluate only the referenced memory unit for this decision.",
                }
                flattened[f"s{index:04d}__{name}"] = batched_question

        started = time.perf_counter()
        if not self.api_key:
            grouped = [self._fallback(memory_unit, questions) for memory_unit, questions in items]
            usage = {"input_tokens": 0, "output_tokens": 0}
        else:
            response = self.session.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "state": {
                        "task": "Classify each referenced memory unit independently.",
                        "memory_units": [
                            {"index": index, "text": memory_unit}
                            for index, (memory_unit, _questions) in enumerate(items)
                        ],
                    },
                    "model": self.model,
                    "questions": flattened,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            flat_answers = payload["answers"]
            usage = payload.get("usage", {})
            grouped = []
            for index, (_, questions) in enumerate(items):
                grouped.append({name: flat_answers.get(f"s{index:04d}__{name}", {}) for name in questions})

        latency = (time.perf_counter() - started) * 1000
        self.request_count += 1
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_latency_ms += latency
        return grouped, usage, latency

    def _fallback(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        lower = state.lower()
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            if question["type"] == "noul":
                value = 0.8 if any(x in lower for x in ("prefer", "project", "current", "now", "latest")) else 0.2
                answers[key] = {"type": "noul", "noul": value}
                continue
            criteria = question["criteria"]
            options = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(criteria))]
            scores = {option: 0.05 for option in options}
            mapping = {
                "preference": ("prefer", "like", "favorite"), "procedure": ("steps", "how to", "procedure"),
                "event": ("yesterday", "went", "met", "happened"), "state": ("currently", "now", "status"),
                "projects": ("project", "build", "work"), "preferences": ("prefer", "like", "favorite"),
                "procedures": ("steps", "how", "recipe"), "environment": ("config", "system", "tool"),
                "episodes": ("yesterday", "today", "met", "went"), "LATEST_STATE": ("latest", "currently", "now"),
                "PREFERENCE": ("prefer", "favorite", "like"), "TEMPORAL": ("when", "before", "after", "date"),
                "MULTI_SESSION": ("across", "over time", "sessions"), "AGGREGATION": ("all", "total", "list"),
            }
            for option in options:
                if any(term in lower for term in mapping.get(option, ())):
                    scores[option] += 0.8
            choice = max(options, key=lambda option: scores[option])
            total = sum(scores.values())
            probs = {option: value / total for option, value in scores.items()}
            answers[key] = {"type": "choice", "choice": choice, "probabilities": probs, "confidence": probs[choice]}
        return answers


class JevMemoryRouter:
    def __init__(
        self, db: MemoryDB, tree: MemoryTree, client: JevClient,
        profile: str = "lean", cache: JevDecisionCache | None = None,
    ):
        self.db, self.tree, self.client = db, tree, client
        self.profile, self.cache = profile, cache
        self.cache_hits = 0
        self.trivial_turns = 0

    @staticmethod
    def _questions(profile: str = "lean") -> dict[str, dict[str, Any]]:
        lean = {
            "priority": {"type": "choice", "instructions": "How valuable is this turn for later retrieval?", "criteria": PRIORITIES},
            "primary_branch": {"type": "choice", "instructions": "Which memory branch best fits this turn?", "criteria": ROOTS},
        }
        if profile == "lean":
            return lean
        return {
            "memory_type": {"type": "choice", "instructions": "What kind of memory is this conversation turn?", "criteria": MEMORY_TYPES},
            **lean,
            "temporal": {"type": "choice", "instructions": "What temporal scope does this turn express?", "criteria": TEMPORAL},
            "create_topic": {"type": "noul", "instructions": "Is there a clear named topic worth a shallow child node?"},
        }

    def route_all(self, case_id: str, show_progress: bool = True, batch_size: int = 25) -> None:
        rows = self.db.unrouted_sentence_rows()
        total = int(self.db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0])
        completed_before = total - len(rows)
        started = time.perf_counter()
        last_width = 0

        def display(done: int, final: bool = False) -> None:
            nonlocal last_width
            elapsed = time.perf_counter() - started
            percent = (done / total * 100) if total else 100.0
            processed = done - completed_before
            average = elapsed / processed if processed else 0.0
            remaining = average * (total - done) if processed else 0.0
            requests = int(getattr(self.client, "request_count", done))
            tokens = int(getattr(self.client, "input_tokens", 0)) + int(getattr(self.client, "output_tokens", 0))
            text = (
                f"Jev routing: {done:,}/{total:,} ({percent:5.1f}%) | "
                f"requests: {requests:,} | cache hits: {self.cache_hits:,} | "
                f"trivial: {self.trivial_turns:,} | tokens: {tokens:,} | "
                f"elapsed: {_duration(elapsed)}"
            )
            if processed and done < total:
                text += f" | ETA: {_duration(remaining)}"
            padding = " " * max(0, last_width - len(text))
            sys.stderr.write("\r" + text + padding + ("\n" if final else ""))
            sys.stderr.flush()
            last_width = len(text)

        if show_progress:
            display(completed_before, final=not rows)
        units_by_turn: dict[int, dict[str, Any]] = {}
        for row in rows:
            unit = units_by_turn.setdefault(
                int(row["turn_id"]), {"content": row["turn_content"], "role": row["role"], "rows": []}
            )
            unit["rows"].append(row)
        units = list(units_by_turn.values())
        batch_size = max(1, int(batch_size))
        completed = completed_before
        misses = []
        # The backend name deliberately distinguishes deterministic smoke-test
        # decisions from live Jev decisions, preventing fallback cache poisoning.
        model = str(getattr(self.client, "backend", getattr(self.client, "model", "unknown")))
        for unit in units:
            questions = self._questions(self.profile)
            if self.profile == "lean" and _is_trivial_acknowledgement(unit["role"], unit["content"]):
                self.trivial_turns += 1
                answers = {
                    "priority": {"type": "choice", "choice": "low", "probabilities": {"low": 1.0}, "confidence": 1.0},
                    "primary_branch": {"type": "choice", "choice": "episodes", "probabilities": {"episodes": 1.0}, "confidence": 1.0},
                }
                for row in unit["rows"]:
                    self._apply_decision(
                        case_id, int(row["id"]), unit["content"], questions, answers, 0.0, 0.0,
                        source="deterministic-trivial",
                    )
                completed += len(unit["rows"])
                continue
            cached = self.cache.get(model, self.profile, unit["role"], unit["content"], questions) if self.cache else None
            if cached is not None:
                self.cache_hits += 1
                for row in unit["rows"]:
                    self._apply_decision(
                        case_id, int(row["id"]), unit["content"], questions, cached, 0.0, 0.0,
                        source="cross-case-cache",
                    )
                completed += len(unit["rows"])
            else:
                unit["questions"] = questions
                unit["state"] = f"Role: {unit['role']}\nContent: {unit['content']}"
                misses.append(unit)
        if show_progress and completed > completed_before:
            display(completed, final=completed == total)

        for offset in range(0, len(misses), batch_size):
            batch = misses[offset:offset + batch_size]
            questions = [unit["questions"] for unit in batch]
            if hasattr(self.client, "ask_batch"):
                grouped, usage, latency = self.client.ask_batch(
                    [(unit["state"], item_questions) for unit, item_questions in zip(batch, questions)]
                )
                sentence_count = sum(len(unit["rows"]) for unit in batch)
                per_sentence_latency = latency / sentence_count
                per_sentence_cost = self.client.cost_for_usage(usage) / sentence_count
                for unit, item_questions, answers in zip(batch, questions, grouped):
                    if self.cache:
                        self.cache.put(model, self.profile, unit["role"], unit["content"], item_questions, answers)
                    for row in unit["rows"]:
                        self._apply_decision(
                            case_id, int(row["id"]), unit["content"], item_questions, answers,
                            per_sentence_latency, per_sentence_cost, source="jev-api",
                        )
            else:
                for unit in batch:
                    answers, usage, latency = self.client.ask(unit["state"], unit["questions"])
                    cost_for_usage = getattr(self.client, "cost_for_usage", lambda _usage: 0.0)
                    if self.cache:
                        self.cache.put(model, self.profile, unit["role"], unit["content"], unit["questions"], answers)
                    for row in unit["rows"]:
                        self._apply_decision(
                            case_id, int(row["id"]), unit["content"], unit["questions"], answers,
                            latency / len(unit["rows"]), cost_for_usage(usage) / len(unit["rows"]), source="jev-api",
                        )
            completed += sum(len(unit["rows"]) for unit in batch)
            if self.cache:
                self.cache.commit()
            if show_progress:
                display(completed, final=completed == total)
        self.db.conn.commit()

    def route(self, case_id: str, sentence_id: int, content: str) -> MemoryDecision:
        questions = self._questions(self.profile)
        answers, usage, latency = self.client.ask(content, questions)
        cost_for_usage = getattr(self.client, "cost_for_usage", lambda _usage: 0.0)
        return self._apply_decision(
            case_id, sentence_id, content, questions, answers, latency, cost_for_usage(usage)
        )

    def _apply_decision(
        self, case_id: str, sentence_id: int, content: str,
        questions: dict[str, dict[str, Any]], answers: dict[str, Any], latency: float, cost: float = 0.0,
        source: str = "jev-api",
    ) -> MemoryDecision:
        try:
            primary = answers["primary_branch"]["choice"]
            branch_probs = answers["primary_branch"].get("probabilities", {})
            alternatives = sorted(
                ((name, float(probability)) for name, probability in branch_probs.items() if name in ROOTS and name != primary),
                key=lambda item: item[1], reverse=True,
            )
            selected = [primary] if primary in ROOTS else []
            if alternatives and alternatives[0][1] >= 0.20:
                selected.append(alternatives[0][0])
            if not selected:
                raise ValueError("no valid branch")
            priority = answers["priority"]["choice"]
            if priority not in PRIORITIES:
                raise ValueError("invalid metadata")
            if self.profile == "full":
                memory_type = answers["memory_type"]["choice"]
                temporal = answers["temporal"]["choice"]
                if memory_type not in MEMORY_TYPES or temporal not in TEMPORAL:
                    raise ValueError("invalid metadata")
                create = float(answers["create_topic"].get("noul", 0)) >= 0.65
            else:
                memory_type = _lean_memory_type(primary)
                temporal = _lean_temporal_scope(content, memory_type)
                create = primary in {"projects", "environment"} and len(re.findall(r"[A-Za-z0-9]+", content)) >= 3
            probabilities = {key: value.get("probabilities", {"yes": value.get("noul")}) for key, value in answers.items()}
            confidence = float(answers["primary_branch"].get("confidence", 0))
            if self.profile == "full":
                confidence = min(confidence, float(answers["memory_type"].get("confidence", 0)))
        except (KeyError, TypeError, ValueError):
            memory_type, priority, temporal, selected, create, confidence = "other", "normal", "unknown", ["episodes"], False, 0.0
            probabilities = {}
        topic = _topic_label(content) if create else None
        assigned = list(selected)
        if create and topic:
            topic_id = self.tree.create_topic(selected[0], topic)
            if topic_id:
                assigned[0] = topic_id
        branch_probs = answers.get("primary_branch", {}).get("probabilities", {})
        for node_id in assigned:
            root = node_id.split("/")[0]
            self.tree.assign(sentence_id, node_id, float(branch_probs.get(root, confidence or 0.5)))
        self.db.conn.execute(
            "UPDATE sentences SET memory_type=?,retrieval_priority=?,temporal_scope=? WHERE id=?",
            (memory_type, priority, temporal, sentence_id),
        )
        action = {"memory_type": memory_type, "retrieval_priority": priority, "temporal_scope": temporal,
                  "selected_node_ids": assigned, "create_topic": bool(create and assigned != selected), "topic": topic,
                  "routing_profile": self.profile, "decision_source": source}
        self.db.log_decision(case_id=case_id, phase="memory-routing", input_hash=hashlib.sha256(content.encode()).hexdigest(),
                             options=questions, action=action, probabilities=probabilities, confidence=confidence,
                             latency_ms=latency, cost=cost)
        return MemoryDecision(memory_type, priority, temporal, assigned, create, topic, confidence, probabilities)
