import json

from src.models import BenchmarkCase
from src.official_eval import (
    OFFICIAL_JUDGE_MODEL,
    OfficialLongMemEvalEvaluator,
    export_hypotheses,
)


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "yes"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }


class _Session:
    def __init__(self):
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return _Response()


def _case(question_id="q1"):
    return BenchmarkCase(
        question_id,
        "single-session-user",
        "What drink?",
        None,
        (),
        "Tea",
        (),
        (),
    )


def test_official_export_has_exact_schema(tmp_path):
    output = tmp_path / "hypotheses.jsonl"
    path, count = export_hypotheses(
        [(_case(), {"mode": "jev-primary", "answer": " Tea. ", "diagnostic": "excluded"})],
        output,
        "jev-primary",
    )
    assert path == output and count == 1
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert entry == {"question_id": "q1", "hypothesis": "Tea."}


def test_official_evaluator_pins_model_and_writes_upstream_log(tmp_path):
    hypotheses = tmp_path / "hypotheses.jsonl"
    hypotheses.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "Tea"}) + "\n",
        encoding="utf-8",
    )
    references = tmp_path / "oracle.json"
    references.write_text(
        json.dumps([{
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What drink?",
            "answer": "Tea",
        }]),
        encoding="utf-8",
    )
    session = _Session()
    evaluator = OfficialLongMemEvalEvaluator("key", session=session)
    result_path, metrics_path, metrics = evaluator.evaluate(hypotheses, references)

    payload = session.requests[0][1]["json"]
    assert payload["model"] == OFFICIAL_JUDGE_MODEL
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 10
    assert payload["n"] == 1
    logged = json.loads(result_path.read_text(encoding="utf-8"))
    assert logged["autoeval_label"] == {"model": OFFICIAL_JUDGE_MODEL, "label": True}
    assert metrics_path.exists()
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["task_averaged_accuracy"] == 1.0
    assert metrics["judge_token_usage"] == {"input": 10, "output": 1}
