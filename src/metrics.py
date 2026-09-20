from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import BenchmarkCase, Evidence


def retrieval_metrics(case: BenchmarkCase, evidence: list[Evidence], context_turn_ids: list[int], db) -> dict:
    relevant_sessions = set(case.answer_session_ids)
    ranked_sessions: list[str] = []
    for item in evidence:
        if item.session_id not in ranked_sessions:
            ranked_sessions.append(item.session_id)
    def recall(k: int) -> float | None:
        return len(relevant_sessions & set(ranked_sessions[:k])) / len(relevant_sessions) if relevant_sessions else None
    first = next((i + 1 for i, sid in enumerate(ranked_sessions) if sid in relevant_sessions), None)
    answer_turns = set(case.answer_turn_keys)
    ranked_turns = []
    for item in evidence:
        row = db.conn.execute("SELECT session_id,turn_index FROM turns WHERE id=?", (item.turn_id,)).fetchone()
        ranked_turns.append((row["session_id"], int(row["turn_index"])))
    def evidence_recall(k: int) -> float | None:
        return len(answer_turns & set(ranked_turns[:k])) / len(answer_turns) if answer_turns else None
    context_turn_keys = set()
    if context_turn_ids:
        marks = ",".join("?" for _ in context_turn_ids)
        for row in db.conn.execute(f"SELECT session_id,turn_index FROM turns WHERE id IN ({marks})", context_turn_ids):
            context_turn_keys.add((row["session_id"], int(row["turn_index"])))
    return {
        "relevant_session_recall_at_5": recall(5), "relevant_session_recall_at_10": recall(10),
        "relevant_evidence_recall_at_10": evidence_recall(10), "relevant_evidence_recall_at_20": evidence_recall(20),
        "mean_reciprocal_rank": 1.0 / first if first else 0.0,
        "precision_at_context_cutoff": (len(answer_turns & context_turn_keys) / len(context_turn_keys)) if context_turn_keys else 0.0,
        "answer_bearing_evidence_present": bool(answer_turns & context_turn_keys) if answer_turns else None,
        "retrieved_candidates": len(evidence), "context_turns": len(context_turn_ids),
    }


def jev_metrics(case: BenchmarkCase, evidence: list[Evidence], plan, db) -> dict:
    all_answer_sentence_ids = set()
    for session_id, turn_index in case.answer_turn_keys:
        rows = db.conn.execute(
            "SELECT s.id FROM sentences s JOIN turns t ON t.id=s.turn_id WHERE t.session_id=? AND t.turn_index=?",
            (session_id, turn_index),
        )
        all_answer_sentence_ids.update(int(row[0]) for row in rows)
    reachable = set()
    for node in plan.selected_node_ids:
        rows = db.conn.execute("SELECT sentence_id FROM node_evidence WHERE node_id=? OR node_id LIKE ?", (node, node + "/%"))
        reachable.update(int(row[0]) for row in rows)
    routing_recall = len(all_answer_sentence_ids & reachable) / len(all_answer_sentence_ids) if all_answer_sentence_ids else None
    decisions = list(db.conn.execute("SELECT confidence,latency_ms,cost FROM jev_decisions WHERE case_id=?", (case.case_id,)))
    return {
        "routing_recall": routing_recall,
        "root_branch_routing_accuracy": routing_recall,
        "topic_routing_recall": routing_recall,
        "answer_evidence_reachable": bool(all_answer_sentence_ids & reachable) if all_answer_sentence_ids else None,
        "evidence_sufficiency_accuracy": None,
        "average_tree_search_depth": sum(1 for step in plan.steps if step["action"] == "SELECT_CHILDREN"),
        "global_rescue_used": plan.global_rescue, "global_rescue_frequency": 1.0 if plan.global_rescue else 0.0,
        "search_steps": len(plan.steps),
        "selected_branches": plan.selected_node_ids,
        "average_decision_confidence": sum(float(row["confidence"]) for row in decisions) / len(decisions) if decisions else None,
        "jev_latency_ms": sum(float(row["latency_ms"]) for row in decisions),
        "jev_cost": sum(float(row["cost"]) for row in decisions),
    }


def aggregate_manual_labels(path: Path) -> dict:
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels: list[int] = []
    for entry in entries:
        value = entry.get("manual_label")
        if isinstance(value, str):
            normalized = value.strip().lower()
            value = 1 if normalized in {"1", "true", "yes", "correct"} else 0 if normalized in {"0", "false", "no", "wrong"} else None
        if value is None:
            response = str(entry.get("manual_response") or "").strip().lower()
            if response in {"yes", "correct"}:
                value = 1
            elif response in {"no", "wrong"}:
                value = 0
        if value in {0, 1, False, True}:
            labels.append(1 if bool(value) else 0)
    return {"judged": len(labels), "pending": len(entries) - len(labels), "correct": sum(labels),
            "accuracy": (sum(labels) / len(labels)) if labels else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-labels", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate_manual_labels(args.manual_labels)
    print(f"Judged prompts: {result['judged']}")
    print(f"Pending prompts: {result['pending']}")
    print(f"Correct: {result['correct']}")
    accuracy = "N/A" if result["accuracy"] is None else f"{result['accuracy'] * 100:.1f}%"
    print(f"Accuracy: {accuracy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
