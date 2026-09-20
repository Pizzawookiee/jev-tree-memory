from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS turns(
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, session_id TEXT NOT NULL,
 session_index INTEGER NOT NULL, turn_index INTEGER NOT NULL, role TEXT NOT NULL,
 content TEXT NOT NULL, session_time TEXT, content_hash TEXT NOT NULL,
 UNIQUE(case_id, session_index, turn_index)
);
CREATE TABLE IF NOT EXISTS sentences(
 id INTEGER PRIMARY KEY, turn_id INTEGER NOT NULL REFERENCES turns(id),
 sentence_index INTEGER NOT NULL, content TEXT NOT NULL,
 start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, embedding BLOB NOT NULL,
 memory_type TEXT, retrieval_priority TEXT, temporal_scope TEXT,
 UNIQUE(turn_id, sentence_index)
);
CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(content, sentence_id UNINDEXED);
CREATE TABLE IF NOT EXISTS tree_nodes(
 id TEXT PRIMARY KEY, parent_id TEXT REFERENCES tree_nodes(id), label TEXT NOT NULL,
 description TEXT NOT NULL, node_type TEXT NOT NULL, depth INTEGER NOT NULL,
 embedding BLOB NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS node_evidence(
 node_id TEXT NOT NULL REFERENCES tree_nodes(id), sentence_id INTEGER NOT NULL REFERENCES sentences(id),
 routing_probability REAL NOT NULL, PRIMARY KEY(node_id, sentence_id)
);
CREATE TABLE IF NOT EXISTS jev_decisions(
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, phase TEXT NOT NULL, input_hash TEXT NOT NULL,
 available_options TEXT NOT NULL, selected_action TEXT NOT NULL, probabilities TEXT NOT NULL,
 confidence REAL NOT NULL, latency_ms REAL NOT NULL, cost REAL NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS runs(
 id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL,
 artifact_json TEXT, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 completed_at TEXT, UNIQUE(case_id, mode)
);
CREATE TABLE IF NOT EXISTS metadata(
 key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS turns_no_update BEFORE UPDATE ON turns BEGIN SELECT RAISE(ABORT, 'turns are immutable'); END;
CREATE TRIGGER IF NOT EXISTS turns_no_delete BEFORE DELETE ON turns BEGIN SELECT RAISE(ABORT, 'turns are immutable'); END;
"""


def pack_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class MemoryDB:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='turns'"
        ).fetchone()
        if row and "session_index, turn_index" not in row["sql"] and "session_index,turn_index" not in row["sql"]:
            self.conn.execute("PRAGMA foreign_keys=OFF")
            self.conn.execute(
                "CREATE TABLE turns_new("
                " id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, session_id TEXT NOT NULL,"
                " session_index INTEGER NOT NULL, turn_index INTEGER NOT NULL, role TEXT NOT NULL,"
                " content TEXT NOT NULL, session_time TEXT, content_hash TEXT NOT NULL,"
                " UNIQUE(case_id, session_index, turn_index)"
                ")"
            )
            self.conn.execute("INSERT INTO turns_new SELECT * FROM turns")
            self.conn.execute("DROP TABLE turns")
            self.conn.execute("ALTER TABLE turns_new RENAME TO turns")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.commit()
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def metadata(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, value)
        )
        self.conn.commit()

    def add_turn(self, values: tuple) -> int:
        cur = self.conn.execute(
            "INSERT INTO turns(case_id,session_id,session_index,turn_index,role,content,session_time,content_hash) VALUES(?,?,?,?,?,?,?,?)",
            values,
        )
        return int(cur.lastrowid)

    def add_sentence(self, turn_id: int, index: int, content: str, start: int, end: int, embedding: np.ndarray) -> int:
        cur = self.conn.execute(
            "INSERT INTO sentences(turn_id,sentence_index,content,start_offset,end_offset,embedding) VALUES(?,?,?,?,?,?)",
            (turn_id, index, content, start, end, pack_vector(embedding)),
        )
        sentence_id = int(cur.lastrowid)
        self.conn.execute("INSERT INTO sentences_fts(content,sentence_id) VALUES(?,?)", (content, sentence_id))
        return sentence_id

    def sentence_rows(self, ids: Iterable[int] | None = None) -> list[sqlite3.Row]:
        base = "SELECT s.*,t.session_id,t.session_time,t.role FROM sentences s JOIN turns t ON t.id=s.turn_id"
        if ids is None:
            return list(self.conn.execute(base))
        values = list(dict.fromkeys(int(x) for x in ids))
        if not values:
            return []
        marks = ",".join("?" for _ in values)
        return list(self.conn.execute(f"{base} WHERE s.id IN ({marks})", values))

    def unrouted_sentence_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT s.*,t.session_id,t.session_time,t.role,t.content AS turn_content "
            "FROM sentences s JOIN turns t ON t.id=s.turn_id "
            "WHERE NOT EXISTS (SELECT 1 FROM node_evidence ne WHERE ne.sentence_id=s.id) "
            "ORDER BY s.id"
        ))

    def log_decision(self, *, case_id: str, phase: str, input_hash: str, options: object,
                     action: object, probabilities: object, confidence: float, latency_ms: float, cost: float = 0.0) -> None:
        self.conn.execute(
            "INSERT INTO jev_decisions(case_id,phase,input_hash,available_options,selected_action,probabilities,confidence,latency_ms,cost) VALUES(?,?,?,?,?,?,?,?,?)",
            (case_id, phase, input_hash, json.dumps(options, sort_keys=True), json.dumps(action, sort_keys=True),
             json.dumps(probabilities, sort_keys=True), confidence, latency_ms, cost),
        )
