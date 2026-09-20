from src.context import expand_and_pack
from src.database import MemoryDB
from src.embeddings import HashEmbedder
from src.ingest import ingest_case
from src.models import BenchmarkCase, Evidence, SearchPlan, TurnInput
from src.search_policy import JevSearchPolicy


def test_context_budget_prioritizes_retrieval_rank_before_chronology(tmp_path):
    case = BenchmarkCase(
        "q", "", "What degree?", None,
        (
            TurnInput("old", 0, 0, "user", "Distractor " * 80, "2020-01-01"),
            TurnInput("answer", 1, 0, "user", "My degree is physics.", "2024-01-01"),
        ),
        None, (), (),
    )
    db = MemoryDB(tmp_path / "context.sqlite")
    ingest_case(db, case, HashEmbedder())
    rows = list(db.conn.execute(
        "SELECT s.id sentence_id,s.turn_id,t.session_id,t.session_time,t.role,s.content "
        "FROM sentences s JOIN turns t ON t.id=s.turn_id ORDER BY t.session_index"
    ))
    distractor, answer = rows[0], rows[-1]
    evidence = [
        Evidence(answer["sentence_id"], answer["turn_id"], answer["session_id"], answer["session_time"], answer["role"], answer["content"]),
        Evidence(distractor["sentence_id"], distractor["turn_id"], distractor["session_id"], distractor["session_time"], distractor["role"], distractor["content"]),
    ]
    packed, turn_ids = expand_and_pack(db, evidence, "POINT_LOOKUP", maximum_budget=30)
    assert "My degree is physics." in packed
    assert answer["turn_id"] in turn_ids
    assert "Distractor" not in packed


def test_insufficient_packed_context_triggers_global_rescue(tmp_path):
    db = MemoryDB(tmp_path / "policy.sqlite")
    class InsufficientJev:
        @staticmethod
        def ask(state, questions):
            return {"sufficient": {"type": "noul", "noul": 0.1}}, {"input_tokens": 100}, 2.0

        @staticmethod
        def cost_for_usage(usage):
            return usage["input_tokens"] / 1_000_000 * 0.042

    client = InsufficientJev()
    policy = JevSearchPolicy(db, None, client)
    plan = SearchPlan("POINT_LOOKUP", ["user"], {"user": 0.8}, False, [])
    updated = policy.sufficiency("q", "What degree?", "Unrelated weather report.", plan)
    assert updated.global_rescue is True
    assert updated.steps[-1]["action"] == "GLOBAL_RESCUE"
    log = db.conn.execute("SELECT cost FROM jev_decisions WHERE phase='evidence-sufficiency'").fetchone()
    assert log is not None and log["cost"] > 0.0
