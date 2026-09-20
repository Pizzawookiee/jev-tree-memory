import sqlite3

import numpy as np
import pytest

from src.database import MemoryDB
from src.embeddings import HashEmbedder
from src.ingest import ingest_case
from src.models import BenchmarkCase, TurnInput


def case():
    return BenchmarkCase(
        "q1", "single-session-user", "What color?", None,
        (TurnInput("s1", 0, 0, "user", "My favorite color is blue. It changed recently.", "2024-01-01"),),
        "ORACLE_SECRET", ("s1",), (("s1", 0),),
    )


def test_immutable_ingestion_and_global_indexes(tmp_path):
    db = MemoryDB(tmp_path / "memory.sqlite")
    ids = ingest_case(db, case(), HashEmbedder())
    turn = db.conn.execute("SELECT * FROM turns").fetchone()
    assert turn["content"] == case().turns[0].content
    assert "ORACLE_SECRET" not in " ".join(row[0] for row in db.conn.execute("SELECT content FROM turns"))
    assert db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0] == len(ids)
    assert db.conn.execute("SELECT count(*) FROM sentences_fts").fetchone()[0] == len(ids)
    for row in db.conn.execute("SELECT s.*,t.content source FROM sentences s JOIN turns t ON t.id=s.turn_id"):
        assert row["source"][row["start_offset"]:row["end_offset"]] == row["content"]
        assert np.frombuffer(row["embedding"], dtype=np.float32).shape == (384,)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("UPDATE turns SET content='changed'")
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute("DELETE FROM turns")


def test_duplicate_session_id_ingestion(tmp_path):
    # LongMemEval cases (e.g. 58bf7951) can contain distinct sessions that share the same session_id.
    dup_case = BenchmarkCase(
        "58bf7951", "multi-session", "What is the goal?", None,
        (
            TurnInput("dup_session", 0, 0, "user", "Session 0 Turn 0", "2023-01-01"),
            TurnInput("dup_session", 0, 1, "assistant", "Session 0 Turn 1", "2023-01-01"),
            TurnInput("dup_session", 1, 0, "user", "Session 1 Turn 0", "2023-01-02"),
            TurnInput("dup_session", 1, 1, "assistant", "Session 1 Turn 1", "2023-01-02"),
        ),
        "secret", ("dup_session",), (("dup_session", 0),),
    )
    db = MemoryDB(tmp_path / "dup.sqlite")
    ids = ingest_case(db, dup_case, HashEmbedder())
    assert len(ids) == 4
    turns = list(db.conn.execute("SELECT session_index, turn_index FROM turns ORDER BY session_index, turn_index"))
    assert len(turns) == 4
    assert turns[0]["session_index"] == 0 and turns[0]["turn_index"] == 0
    assert turns[2]["session_index"] == 1 and turns[2]["turn_index"] == 0


def test_schema_migration_from_legacy_unique_constraint(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE turns(
         id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, session_id TEXT NOT NULL,
         session_index INTEGER NOT NULL, turn_index INTEGER NOT NULL, role TEXT NOT NULL,
         content TEXT NOT NULL, session_time TEXT, content_hash TEXT NOT NULL,
         UNIQUE(case_id, session_id, turn_index)
        );
    """)
    conn.close()

    # Opening with MemoryDB should automatically migrate to UNIQUE(case_id, session_index, turn_index)
    db = MemoryDB(db_path)
    row = db.conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='turns'").fetchone()
    assert "session_index, turn_index" in row[0]
    db.close()


