from __future__ import annotations

import re
from uuid import uuid5, NAMESPACE_URL

from .database import MemoryDB, pack_vector
from .embeddings import Embedder

ROOTS = {
    "user": "Stable facts about the user and their identity or background",
    "projects": "Named projects, work streams, goals, and tasks",
    "preferences": "Likes, dislikes, habits, choices, and personal preferences",
    "procedures": "Instructions, methods, recipes, and repeatable processes",
    "environment": "Tools, systems, configuration, locations, and current state",
    "episodes": "Events, conversations, and time-bounded experiences",
}


class MemoryTree:
    def __init__(self, db: MemoryDB, embedder: Embedder, max_depth: int = 3):
        self.db = db
        self.embedder = embedder
        self.max_depth = max_depth

    def initialize(self) -> None:
        if self.db.conn.execute("SELECT 1 FROM tree_nodes WHERE id='root'").fetchone():
            return
        values = [("root", None, "root", "All recoverable memory evidence", "root", 0)]
        values.extend((name, "root", name, desc, "branch", 1) for name, desc in ROOTS.items())
        vectors = self.embedder.encode([f"{v[2]}: {v[3]}" for v in values])
        for value, vector in zip(values, vectors):
            self.db.conn.execute(
                "INSERT OR IGNORE INTO tree_nodes(id,parent_id,label,description,node_type,depth,embedding) VALUES(?,?,?,?,?,?,?)",
                (*value, pack_vector(vector)),
            )
        self.db.conn.commit()

    def children(self, parent_id: str) -> list[dict]:
        return [dict(row) for row in self.db.conn.execute("SELECT * FROM tree_nodes WHERE parent_id=? ORDER BY id", (parent_id,))]

    def create_topic(self, parent_id: str, label: str) -> str | None:
        parent = self.db.conn.execute("SELECT depth FROM tree_nodes WHERE id=?", (parent_id,)).fetchone()
        if not parent or parent["depth"] >= self.max_depth:
            return None
        clean = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:48] or "topic"
        node_id = f"{parent_id}/{clean}-{str(uuid5(NAMESPACE_URL, parent_id + ':' + clean))[:8]}"
        if self.db.conn.execute("SELECT 1 FROM tree_nodes WHERE id=?", (node_id,)).fetchone():
            return node_id
        vector = self.embedder.encode([f"{clean}: topic memory"])[0]
        self.db.conn.execute(
            "INSERT OR IGNORE INTO tree_nodes(id,parent_id,label,description,node_type,depth,embedding) VALUES(?,?,?,?,?,?,?)",
            (node_id, parent_id, clean, f"Topic evidence about {clean}", "topic", parent["depth"] + 1, pack_vector(vector)),
        )
        return node_id

    def assign(self, sentence_id: int, node_id: str, probability: float) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO node_evidence(node_id,sentence_id,routing_probability) VALUES(?,?,?)",
            (node_id, sentence_id, max(0.0, min(1.0, probability))),
        )

    def evidence_ids(self, node_ids: list[str]) -> list[int]:
        if not node_ids:
            return []
        clauses = " OR ".join("node_id=? OR node_id LIKE ?" for _ in node_ids)
        args = [item for node in node_ids for item in (node, node + "/%")]
        return [int(row[0]) for row in self.db.conn.execute(
            f"SELECT DISTINCT sentence_id FROM node_evidence WHERE {clauses}", args
        )]
