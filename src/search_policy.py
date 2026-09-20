from __future__ import annotations

import hashlib

from .database import MemoryDB
from .jev import JevClient
from .models import SearchPlan
from .tree import ROOTS, MemoryTree

QUERY_TYPES = {
    "POINT_LOOKUP": "One specific fact or answer",
    "PREFERENCE": "A like, dislike, habit, or preference",
    "LATEST_STATE": "The most recent value or current status",
    "TEMPORAL": "Dates, ordering, duration, before, or after",
    "MULTI_SESSION": "Requires evidence from multiple sessions",
    "AGGREGATION": "Requires collecting or counting several facts",
}


class JevSearchPolicy:
    def __init__(self, db: MemoryDB, tree: MemoryTree, client: JevClient, max_steps: int = 6):
        self.db, self.tree, self.client, self.max_steps = db, tree, client, max_steps

    def plan(self, case_id: str, question: str) -> SearchPlan:
        questions = {
            "query_type": {"type": "choice", "instructions": "What retrieval pattern does this question require?", "criteria": QUERY_TYPES},
            "primary_branch": {"type": "choice", "instructions": "Which memory branch is most likely to contain the answer?", "criteria": ROOTS},
            "secondary_branch": {"type": "choice", "instructions": "Which additional branch is most useful?", "criteria": {**ROOTS, "none": "No additional branch"}},
        }
        answers, usage, latency = self.client.ask(question, questions)
        try:
            query_type = answers["query_type"]["choice"]
            if query_type not in QUERY_TYPES:
                raise ValueError
            primary = answers["primary_branch"]["choice"]
            secondary = answers["secondary_branch"]["choice"]
            selected = [primary] + ([secondary] if secondary != "none" and secondary != primary else [])
            selected = [node for node in selected if node in ROOTS][:2]
            if not selected:
                raise ValueError
            probabilities = {k: float(v) for k, v in answers["primary_branch"].get("probabilities", {}).items() if k in ROOTS}
            confidence = float(answers["primary_branch"].get("confidence", 0))
        except (KeyError, TypeError, ValueError):
            query_type, selected, probabilities, confidence = "POINT_LOOKUP", ["user", "episodes"], {"user": 0.5, "episodes": 0.5}, 0.0
        action = {"action": "SELECT_CHILDREN", "selected_child_ids": selected, "candidate_budget": 40}
        self.db.log_decision(case_id=case_id, phase="query-routing", input_hash=hashlib.sha256(question.encode()).hexdigest(),
                             options=questions, action={"query_type": query_type, **action}, probabilities=probabilities,
                             confidence=confidence, latency_ms=latency, cost=self.client.cost_for_usage(usage))
        return SearchPlan(query_type, selected, probabilities, False, [action])

    def sufficiency(self, case_id: str, question: str, packed_context: str, plan: SearchPlan) -> SearchPlan:
        remaining = [name for name in ROOTS if name not in plan.selected_node_ids]
        state = (
            f"Question: {question}\nSearched: {plan.selected_node_ids}\n"
            f"Actual context that will be sent to the answerer:\n{packed_context or '(none)'}\n"
            f"Remaining branches: {remaining}"
        )
        questions = {
            "sufficient": {"type": "noul", "instructions": "Does the actual answerer context contain enough information to answer the question correctly?"},
        }
        answers, usage, latency = self.client.ask(state, questions)
        score = float(answers.get("sufficient", {}).get("noul", 0))
        if score >= 0.65:
            action = {"action": "STOP", "selected_child_ids": [], "candidate_budget": 0}
        else:
            plan.global_rescue = True
            action = {"action": "GLOBAL_RESCUE", "selected_child_ids": [], "candidate_budget": 50}
        plan.steps.append(action)
        self.db.log_decision(case_id=case_id, phase="evidence-sufficiency", input_hash=hashlib.sha256(state.encode()).hexdigest(),
                             options=questions, action=action, probabilities={"sufficient": score}, confidence=abs(score - 0.5) * 2,
                             latency_ms=latency, cost=self.client.cost_for_usage(usage))
        return plan
