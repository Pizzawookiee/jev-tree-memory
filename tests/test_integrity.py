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

