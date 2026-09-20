import json

from src.database import MemoryDB
from src.embeddings import HashEmbedder
from src.ingest import ingest_case
from src.jev import JevClient, JevMemoryRouter
from src.jev_cache import JevDecisionCache
from src.models import BenchmarkCase, TurnInput
from src.tree import MemoryTree


class InvalidJev:
    backend = "invalid-test"
    def ask(self, state, questions):
        return {"bad": {}}, {}, 1.0


def test_invalid_decision_falls_back_without_suppressing_evidence(tmp_path):
    case = BenchmarkCase("q", "", "q", None, (TurnInput("s", 0, 0, "user", "Evidence remains.", None),), None, (), ())
    db = MemoryDB(tmp_path / "db.sqlite")
    ingest_case(db, case, HashEmbedder())
    tree = MemoryTree(db, HashEmbedder())
    tree.initialize()
    JevMemoryRouter(db, tree, InvalidJev()).route_all("q")
    assert db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0] == 1
    assert db.conn.execute("SELECT count(*) FROM sentences_fts").fetchone()[0] == 1
    sentence = db.conn.execute("SELECT * FROM sentences").fetchone()
    assert (sentence["memory_type"], sentence["retrieval_priority"], sentence["temporal_scope"]) == ("other", "normal", "unknown")
    assert db.conn.execute("SELECT node_id FROM node_evidence").fetchone()[0] == "episodes"
    log = db.conn.execute("SELECT * FROM jev_decisions").fetchone()
    assert log["selected_action"] and log["probabilities"]


def test_tree_depth_limit(tmp_path):
    db = MemoryDB(tmp_path / "db.sqlite")
    tree = MemoryTree(db, HashEmbedder(), max_depth=2)
    tree.initialize()
    child = tree.create_topic("projects", "alpha")
    assert child
    assert tree.create_topic(child, "too-deep") is None


def test_sentence_routing_batches_jev_requests(tmp_path):
    turns = (TurnInput("s", 0, 0, "user", "One fact. Two facts. Three facts. Four facts. Five facts.", None),)
    case = BenchmarkCase("q", "", "q", None, turns, None, (), ())
    db = MemoryDB(tmp_path / "batch.sqlite")
    ingest_case(db, case, HashEmbedder())
    tree = MemoryTree(db, HashEmbedder())
    tree.initialize()
    client = JevClient(None, allow_fallback=True)
    JevMemoryRouter(db, tree, client).route_all("q", show_progress=False, batch_size=2)
    # All sentences from one raw turn share one Jev classification request.
    assert client.request_count == 1
    count = db.conn.execute("SELECT count(*) FROM jev_decisions WHERE phase='memory-routing'").fetchone()[0]
    assert count == 5

    # A repeated or resumed pass reuses all completed routes without new calls.
    JevMemoryRouter(db, tree, client).route_all("q", show_progress=False, batch_size=2)
    assert client.request_count == 1


def test_batch_payload_places_each_memory_unit_in_shared_state_once():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            answers = {}
            for key, question in self.payload["questions"].items():
                if question["type"] == "noul":
                    answers[key] = {"type": "noul", "noul": 0.2}
                else:
                    options = list(question["criteria"])
                    answers[key] = {
                        "type": "choice", "choice": options[0],
                        "probabilities": {option: (1.0 if option == options[0] else 0.0) for option in options},
                        "confidence": 1.0,
                    }
            return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 2}}

    class Session:
        payload = None

        def post(self, *args, **kwargs):
            self.payload = kwargs["json"]
            return Response(self.payload)

    session = Session()
    client = JevClient("test-key", session=session)
    questions = JevMemoryRouter._questions()
    grouped, usage, _latency = client.ask_batch([("First turn text", questions), ("Second turn text", questions)])
    serialized = json.dumps(session.payload)
    assert serialized.count("First turn text") == 1
    assert serialized.count("Second turn text") == 1
    assert len(grouped) == 2 and usage["input_tokens"] == 10
    assert client.request_count == 1 and client.estimated_cost > 0


def test_lean_profile_filters_only_exact_assistant_acknowledgements(tmp_path):
    case = BenchmarkCase(
        "q", "", "q", None,
        (TurnInput("s", 0, 0, "assistant", "Got it.", None),), None, (), (),
    )
    db = MemoryDB(tmp_path / "trivial.sqlite")
    ingest_case(db, case, HashEmbedder())
    tree = MemoryTree(db, HashEmbedder())
    tree.initialize()
    client = JevClient(None, allow_fallback=True)
    router = JevMemoryRouter(db, tree, client, profile="lean")
    router.route_all("q", show_progress=False)
    assert client.request_count == 0
    assert router.trivial_turns == 1
    action = json.loads(db.conn.execute("SELECT selected_action FROM jev_decisions").fetchone()[0])
    assert action["decision_source"] == "deterministic-trivial"
    assert action["routing_profile"] == "lean"


def test_full_diagnostics_bypasses_trivial_filter(tmp_path):
    case = BenchmarkCase(
        "q", "", "q", None,
        (TurnInput("s", 0, 0, "assistant", "Got it.", None),), None, (), (),
    )
    db = MemoryDB(tmp_path / "full.sqlite")
    ingest_case(db, case, HashEmbedder())
    tree = MemoryTree(db, HashEmbedder())
    tree.initialize()
    client = JevClient(None, allow_fallback=True)
    router = JevMemoryRouter(db, tree, client, profile="full")
    router.route_all("q", show_progress=False)
    assert client.request_count == 1
    assert router.trivial_turns == 0
    assert set(router._questions("full")) == {
        "memory_type", "priority", "primary_branch", "temporal", "create_topic"
    }


def test_cross_case_cache_reuses_same_role_and_content(tmp_path):
    cache = JevDecisionCache(tmp_path / "routing-cache.sqlite")
    case = BenchmarkCase(
        "q", "", "q", None,
        (TurnInput("s", 0, 0, "user", "I prefer green tea.", None),), None, (), (),
    )

    first_db = MemoryDB(tmp_path / "first.sqlite")
    ingest_case(first_db, case, HashEmbedder())
    first_tree = MemoryTree(first_db, HashEmbedder())
    first_tree.initialize()
    first_client = JevClient(None, allow_fallback=True)
    JevMemoryRouter(first_db, first_tree, first_client, profile="lean", cache=cache).route_all(
        "q", show_progress=False
    )
    assert first_client.request_count == 1

    second_db = MemoryDB(tmp_path / "second.sqlite")
    ingest_case(second_db, case, HashEmbedder())
    second_tree = MemoryTree(second_db, HashEmbedder())
    second_tree.initialize()
    second_client = JevClient(None, allow_fallback=True)
    second_router = JevMemoryRouter(second_db, second_tree, second_client, profile="lean", cache=cache)
    second_router.route_all("q", show_progress=False)
    assert second_client.request_count == 0
    assert second_router.cache_hits == 1
    action = json.loads(second_db.conn.execute("SELECT selected_action FROM jev_decisions").fetchone()[0])
    assert action["decision_source"] == "cross-case-cache"
    cache.close()
