from __future__ import annotations

import hashlib

from .database import MemoryDB
from .embeddings import Embedder
from .models import BenchmarkCase
from .split import split_sentences


def ingest_case(db: MemoryDB, case: BenchmarkCase, embedder: Embedder, batch_size: int = 64) -> list[int]:
    pending: list[tuple[int, int, str, int, int]] = []
    for turn in case.turns:
        if turn.role not in {"user", "assistant", "system", "tool"} or not isinstance(turn.content, str):
            raise ValueError(f"Invalid turn {turn.session_id}/{turn.turn_index}")
        content_hash = hashlib.sha256(turn.content.encode("utf-8")).hexdigest()
        turn_id = db.add_turn((case.case_id, turn.session_id, turn.session_index, turn.turn_index,
                               turn.role, turn.content, turn.session_time, content_hash))
        spans = split_sentences(turn.content)
        for sentence_index, (content, start, end) in enumerate(spans):
            if turn.content[start:end] != content:
                raise AssertionError("sentence offsets do not reconstruct source text")
            pending.append((turn_id, sentence_index, content, start, end))
    # SentenceTransformer performs its own internal batching. One encode call
    # avoids Python/model setup overhead for every small outer chunk.
    vectors = embedder.encode([item[2] for item in pending]) if pending else []
    ids: list[int] = []
    for item, vector in zip(pending, vectors):
        ids.append(db.add_sentence(*item, vector))
    db.conn.commit()
    return ids
