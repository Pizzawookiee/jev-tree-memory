import json

from src import runner


def test_compare_retrieval_only_and_resume(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([{
        "question_id": "real-shape-1", "question_type": "single-session-preference",
        "question": "What drink do I prefer?", "answer": "tea", "question_date": "2024-02-01",
        "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2024-01-01", "2024-01-02"],
        "haystack_sessions": [
            [{"role": "user", "content": "I prefer tea to coffee.", "has_answer": True}, {"role": "assistant", "content": "Noted."}],
            [{"role": "user", "content": "The weather is mild."}, {"role": "assistant", "content": "Yes."}],
        ],
        "answer_session_ids": ["s1"],
    }]), encoding="utf-8")
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    code = runner.main(["--compare", "--dataset", str(dataset), "--retrieval-only", "--trace-jev", "--allow-model-fallback"])
    assert code == 0
    baseline = json.loads((tmp_path / "results" / "real-shape-1" / "baseline.json").read_text())
    jev = json.loads((tmp_path / "results" / "real-shape-1" / "jev-primary.json").read_text())
    assert baseline["backends"]["jev"] is None
    assert "jev_trace" not in baseline
    assert jev["jev_trace"]
    assert baseline["backends"]["reranker"] == jev["backends"]["reranker"]
    assert baseline["retrieved_evidence"] and jev["retrieved_evidence"]
    progress = capsys.readouterr().err
    assert "Starting baseline retrieval" in progress
    assert "Loading embedding model..." in progress
    assert "Splitting and embedding" in progress
    assert "Ingestion complete:" in progress
    assert "Jev routing:" in progress
    assert "requests:" in progress
    assert "Jev query decision:" in progress
    assert "Jev sufficiency decision:" in progress
    assert "Jev complete:" in progress
    assert runner.main(["--compare", "--dataset", str(dataset), "--retrieval-only", "--resume", "--allow-model-fallback"]) == 0


def test_answer_only_writes_official_hypothesis_file(tmp_path, monkeypatch):
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([{
        "question_id": "official-1", "question_type": "single-session-user",
        "question": "What drink?", "answer": "tea", "question_date": "2024-02-01",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "I drink tea.", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }]), encoding="utf-8")
    results = tmp_path / "results"
    output = results / "submission.jsonl"
    monkeypatch.setenv("RESULTS_DIR", str(results))
    assert runner.main([
        "--mode", "jev-primary", "--dataset", str(dataset), "--answer-only",
        "--hypothesis-output", str(output), "--allow-model-fallback",
    ]) == 0
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert set(entry) == {"question_id", "hypothesis"}
    assert entry["question_id"] == "official-1"
