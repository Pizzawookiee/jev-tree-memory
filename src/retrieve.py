from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .database import MemoryDB, unpack_vector
from .embeddings import Embedder
from .models import Evidence, SearchPlan
from .tree import MemoryTree


def _evidence(row) -> Evidence:
    priority = {"low": 0.0, "normal": 0.5, "high": 1.0}.get(row["retrieval_priority"], 0.5)
    return Evidence(int(row["id"]), int(row["turn_id"]), row["session_id"], row["session_time"], row["role"], row["content"], priority_score=priority)


def _vector_scores(db: MemoryDB, query: np.ndarray, ids: list[int] | None, top_k: int) -> list[tuple[int, float]]:
    rows = db.sentence_rows(ids)
    scored = [(int(row["id"]), float(np.dot(query, unpack_vector(row["embedding"])))) for row in rows]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def _fts_query(question: str) -> str:
    terms = re.findall(r"[A-Za-z0-9]{2,}", question)
    return " OR ".join(f'"{term}"' for term in terms[:30]) or '"emptyquery"'


def _bm25_scores(db: MemoryDB, question: str, ids: list[int] | None, top_k: int) -> list[tuple[int, float]]:
    try:
        rows = db.conn.execute(
            "SELECT CAST(sentence_id AS INTEGER) sid,bm25(sentences_fts) rank FROM sentences_fts WHERE sentences_fts MATCH ? ORDER BY rank LIMIT ?",
            (_fts_query(question), max(top_k * 5, 50)),
        )
    except Exception:
        return []
    allowed = set(ids) if ids is not None else None
    raw = [(int(row["sid"]), float(row["rank"])) for row in rows if allowed is None or int(row["sid"]) in allowed][:top_k]
    if not raw:
        return []
    values = [-rank for _, rank in raw]
    lo, hi = min(values), max(values)
    return [(sid, (value - lo) / (hi - lo) if hi > lo else 1.0) for (sid, _), value in zip(raw, values)]


class Retriever:
    def __init__(self, db: MemoryDB, embedder: Embedder, tree: MemoryTree):
        self.db, self.embedder, self.tree = db, embedder, tree

    def baseline(self, question: str, top_k: int = 50) -> list[Evidence]:
        query = self.embedder.encode_query([question])[0]
        vector = _vector_scores(self.db, query, None, top_k)
        bm25 = _bm25_scores(self.db, question, None, top_k)
        return self._fuse(vector, bm25, {}, {}, mode="baseline", top_k=top_k)

    def jev(self, question: str, plan: SearchPlan, top_k: int = 50) -> list[Evidence]:
        query = self.embedder.encode_query([question])[0]
        branch_ids = self.tree.evidence_ids(plan.selected_node_ids)
        branch_budget = max(1, math.floor(top_k * 0.8))
        rescue_budget = max(1, top_k - branch_budget)
        branch_v = _vector_scores(self.db, query, branch_ids, top_k)
        branch_b = _bm25_scores(self.db, question, branch_ids, top_k)
        global_v = _vector_scores(self.db, query, None, top_k)
        global_b = _bm25_scores(self.db, question, None, top_k)
        jev_scores: dict[int, float] = defaultdict(float)
        node_map: dict[int, list[str]] = defaultdict(list)
        for row in self.db.conn.execute("SELECT node_id,sentence_id,routing_probability FROM node_evidence"):
            root = row["node_id"].split("/")[0]
            if root in plan.selected_node_ids or any(row["node_id"].startswith(node + "/") for node in plan.selected_node_ids):
                score = plan.branch_probabilities.get(root, 0.25) * float(row["routing_probability"])
                jev_scores[int(row["sentence_id"])] = max(jev_scores[int(row["sentence_id"])], score)
                node_map[int(row["sentence_id"])].append(row["node_id"])
        if plan.global_rescue:
            # A failed sufficiency check removes the 80/20 branch allocation
            # and its branch-score bias, then searches the full corpus.
            return self._fuse(global_v, global_b, {}, node_map, mode="baseline", top_k=top_k)
        branch = self._fuse(branch_v, branch_b, jev_scores, node_map, mode="jev-primary", top_k=branch_budget)
        branch_sentence_ids = {item.sentence_id for item in branch}
        global_ranked = self._fuse(global_v, global_b, jev_scores, node_map, mode="jev-primary", top_k=top_k)
        rescue = [item for item in global_ranked if item.sentence_id not in branch_sentence_ids][:rescue_budget]
        # Candidate allocation is therefore at most 80% selected-tree and 20%
        # global rescue, before the identical cross-encoder stage.
        return sorted(branch + rescue, key=lambda item: item.hybrid_score, reverse=True)

    def _fuse(self, vector, bm25, jev_scores, node_map, *, mode: str, top_k: int) -> list[Evidence]:
        vector_map, bm25_map = dict(vector), dict(bm25)
        ids = set(vector_map) | set(bm25_map)
        rows = {int(row["id"]): row for row in self.db.sentence_rows(ids)}
        results: list[Evidence] = []
        for sentence_id in ids:
            item = _evidence(rows[sentence_id])
            item.vector_score = vector_map.get(sentence_id, 0.0)
            item.bm25_score = bm25_map.get(sentence_id, 0.0)
            item.jev_score = jev_scores.get(sentence_id, 0.0)
            item.node_ids = node_map.get(sentence_id, [])
            item.hybrid_score = (0.60 * item.vector_score + 0.40 * item.bm25_score) if mode == "baseline" else (
                0.25 * item.vector_score + 0.15 * item.bm25_score + 0.50 * item.jev_score + 0.10 * item.priority_score)
            results.append(item)
        return sorted(results, key=lambda item: item.hybrid_score, reverse=True)[:top_k]
