import json
import re
from types import SimpleNamespace

import pytest

from src import runner
from src.judge import OpenAIBatchJudge, OpenAIJudge
from src.judge_prompt import build_judge_prompt
from src.manual_answer import export_manual_answer_prompts
from src.manual_judge import export_manual_prompts, prediction_hash
from src.metrics import aggregate_manual_labels
from src.models import BenchmarkCase


def _dataset(path):
    path.write_text(json.dumps([{
        "question_id": "case_001", "question_type": "single-session-user",
        "question": "What drink?", "answer": "Tea", "question_date": "2024-02-01",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "I drink tea.", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }]), encoding="utf-8")


@pytest.mark.parametrize("arguments", [
    ["--compare", "--manual-judge", "--batch"],
    ["--compare", "--manual-judge", "--retrieval-only"],
    ["--compare", "--case-id", "x", "--all"],
    ["--mode", "baseline", "--compare"],
    ["--compare", "--retrieval-only", "--judge-only"],
    ["--compare", "--manual-answer", "--manual-judge"],
    ["--compare", "--manual-answer-input", "answers.jsonl"],
    ["--compare", "--hypothesis-output", "hypotheses.jsonl"],
])
def test_incompatible_cli_combinations_are_rejected(arguments):
    with pytest.raises(SystemExit):
        runner.parse_args(arguments)


def test_prompt_parity_for_manual_sync_and_batch():
    prompt = build_judge_prompt("case_001", "Question?", "Answer", "Hypothesis",
                                question_type="single-session-user")
    sync = OpenAIJudge("test-key", "gpt-4o")
    batch = OpenAIBatchJudge("test-key", "gpt-4o")
    rebuilt = sync.prompt("case_001", "single-session-user", "Question?", "Answer", "Hypothesis")
    assert rebuilt == prompt
    assert sync.request(prompt) == batch.request(prompt, "case_001:baseline")["body"]


def test_manual_compare_custom_path_and_no_judge_call(tmp_path, monkeypatch):
    dataset = tmp_path / "data.json"
    _dataset(dataset)
    results = tmp_path / "results"
    output = results / "custom.jsonl"
    monkeypatch.setenv("RESULTS_DIR", str(results))

    def fake_answer(case, artifact, settings, args):
        artifact["answer"] = "Tea"
        artifact["backends"]["answerer"] = "fake-answerer"

    monkeypatch.setattr(runner, "_generate_answer", fake_answer)
    assert runner.main(["--compare", "--limit", "1", "--manual-judge", "--manual-judge-output", str(output),
                        "--dataset", str(dataset), "--allow-model-fallback"]) == 0
    entries = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {entry["manual_judge_id"] for entry in entries} == {"case_001:baseline", "case_001:jev-primary"}
    assert len({entry["prediction_hash"] for entry in entries}) == 1
    assert output.with_suffix(".md").exists()
    assert all(entry["expected_response_format"] == "yes or no" for entry in entries)

    # Resume regenerates the authoritative file without duplicate entries.
    assert runner.main(["--compare", "--limit", "1", "--manual-judge", "--manual-judge-output", str(output),
                        "--dataset", str(dataset), "--allow-model-fallback", "--resume"]) == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_labels_survive_same_prediction_and_invalidate_changed_prediction(tmp_path):
    case = BenchmarkCase("case_001", "single-session-user", "Question?", None, (), "Answer", (), ())
    path = tmp_path / "prompts.jsonl"
    artifact = {"mode": "baseline", "answer": "first"}
    export_manual_prompts([(case, artifact)], path, "gpt-4o")
    entry = json.loads(path.read_text())
    entry.update({"manual_response": "CORRECT", "manual_label": 1})
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    export_manual_prompts([(case, artifact)], path, "gpt-4o")
    same = json.loads(path.read_text())
    assert same["manual_label"] == 1
    artifact["answer"] = "changed"
    export_manual_prompts([(case, artifact)], path, "gpt-4o")
    changed = json.loads(path.read_text())
    assert changed["prediction_hash"] == prediction_hash("changed")
    assert changed["manual_label"] is None and changed["manual_response"] is None


def test_manual_answer_export_and_import_without_answer_api(tmp_path, monkeypatch):
    dataset = tmp_path / "data.json"
    _dataset(dataset)
    results = tmp_path / "results"
    answer_path = results / "answers.jsonl"
    judge_path = results / "judges.jsonl"
    monkeypatch.setenv("RESULTS_DIR", str(results))

    class ForbiddenAnswerer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("manual answer workflow instantiated the API answerer")

    monkeypatch.setattr(runner, "OpenAIAnswerer", ForbiddenAnswerer)
    assert runner.main(["--compare", "--limit", "1", "--manual-answer",
                        "--manual-answer-output", str(answer_path), "--dataset", str(dataset),
                        "--allow-model-fallback"]) == 0
    entries = [json.loads(line) for line in answer_path.read_text(encoding="utf-8").splitlines()]
    assert {entry["manual_answer_id"] for entry in entries} == {"case_001:baseline", "case_001:jev-primary"}
    assert all(entry["suggested_answer"] is None for entry in entries)
    assert all("Tea" not in str(entry) for entry in entries)
    for entry in entries:
        entry["suggested_answer"] = "Tea"
    answer_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")

    assert runner.main(["--compare", "--limit", "1", "--manual-judge",
                        "--manual-answer-input", str(answer_path), "--manual-judge-output", str(judge_path),
                        "--dataset", str(dataset), "--allow-model-fallback", "--resume"]) == 0
    judge_entries = [json.loads(line) for line in judge_path.read_text(encoding="utf-8").splitlines()]
    assert len(judge_entries) == 2
    for mode in ("baseline", "jev-primary"):
        artifact = json.loads((results / "case_001" / f"{mode}.json").read_text(encoding="utf-8"))
        assert artifact["answer"] == "Tea"
        assert artifact["backends"]["answerer"] == "manual-chatgpt-import"
        assert artifact["answer_api_called"] is False


def test_manual_label_aggregation_excludes_pending(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in [
        {"manual_response": "CORRECT", "manual_label": 1},
        {"manual_response": "WRONG", "manual_label": 0},
        {"manual_response": None, "manual_label": None},
    ]) + "\n", encoding="utf-8")
    assert aggregate_manual_labels(path) == {"judged": 2, "pending": 1, "correct": 1, "accuracy": 0.5}


def test_root_batch_script_contains_active_smoke_test():
    text = (runner.Path(__file__).parents[1] / "run_tests.bat").read_text(encoding="utf-8")
    active_commands = [line.strip() for line in text.splitlines() if line.strip().lower().startswith("python ")]
    assert "python -m pytest tests -q" in active_commands
    assert any(
        command.startswith("python -m src.runner")
        and bool(re.search(r"--limit\s+\d+", command))
        and ("--compare" in command or "--mode jev-primary" in command)
        for command in active_commands
    )
    assert "if errorlevel 1 goto :error" in text
