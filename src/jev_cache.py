from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


class JevDecisionCache:
    """Persistent model-decision cache; contains no benchmark oracle data."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS routing_decisions(
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                profile TEXT NOT NULL,
                role TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                question_schema_hash TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    @staticmethod
    def key(model: str, profile: str, role: str, content: str, questions: dict[str, Any]) -> tuple[str, str, str]:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        schema = json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        schema_hash = hashlib.sha256(schema.encode("utf-8")).hexdigest()
        payload = json.dumps(
            {"model": model, "profile": profile, "role": role, "content_hash": content_hash,
             "question_schema_hash": schema_hash},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), content_hash, schema_hash

    def get(self, model: str, profile: str, role: str, content: str, questions: dict[str, Any]) -> dict | None:
        cache_key, _content_hash, _schema_hash = self.key(model, profile, role, content, questions)
        row = self.conn.execute(
            "SELECT answers_json FROM routing_decisions WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, model: str, profile: str, role: str, content: str,
        questions: dict[str, Any], answers: dict[str, Any],
    ) -> None:
        cache_key, content_hash, schema_hash = self.key(model, profile, role, content, questions)
        self.conn.execute(
            "INSERT OR REPLACE INTO routing_decisions"
            "(cache_key,model,profile,role,content_hash,question_schema_hash,answers_json) VALUES(?,?,?,?,?,?,?)",
            (cache_key, model, profile, role, content_hash, schema_hash,
             json.dumps(answers, sort_keys=True, ensure_ascii=False)),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
