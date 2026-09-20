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


def test_clean_db_removes_sqlite_database(tmp_path, monkeypatch):
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([{
        "question_id": "clean-1", "question_type": "single-session-user",
        "question": "What drink?", "answer": "tea", "question_date": "2024-02-01",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "I drink tea.", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }]), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setenv("RESULTS_DIR", str(results))
    assert runner.main([
        "--mode", "baseline", "--dataset", str(dataset), "--retrieval-only",
        "--clean-db", "--allow-model-fallback",
    ]) == 0
    assert (results / "clean-1" / "baseline.json").exists()
    assert not (results / "databases" / "clean-1.sqlite").exists()


def test_jev_trace_compact_vs_full_diagnostics(tmp_path, monkeypatch):
    dataset = tmp_path / "longmemeval.json"
    dataset.write_text(json.dumps([{
        "question_id": "diag-1", "question_type": "single-session-user",
        "question": "What drink?", "answer": "tea", "question_date": "2024-02-01",
        "haystack_session_ids": ["s1"], "haystack_dates": ["2024-01-01"],
        "haystack_sessions": [[{"role": "user", "content": "Fact one. Fact two. Fact three.", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    }]), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setenv("RESULTS_DIR", str(results))
    # Compact run
    assert runner.main([
        "--mode", "jev-primary", "--dataset", str(dataset), "--retrieval-only",
        "--allow-model-fallback",
    ]) == 0
    compact_jev = json.loads((results / "diag-1" / "jev-primary.json").read_text())
    assert compact_jev["jev_trace"]
    # All rows in compact trace are search decisions (or non-memory-routing)
    assert all(step.get("phase") != "memory-routing" for step in compact_jev["jev_trace"])

    # Full diagnostics run
    results_full = tmp_path / "results_full"
    monkeypatch.setenv("RESULTS_DIR", str(results_full))
    assert runner.main([
        "--mode", "jev-primary", "--dataset", str(dataset), "--retrieval-only",
        "--full-jev-diagnostics", "--allow-model-fallback",
    ]) == 0
    full_jev = json.loads((results_full / "diag-1" / "jev-primary.json").read_text())
    assert any(step.get("phase") == "memory-routing" for step in full_jev["jev_trace"])
    assert len(full_jev["jev_trace"]) > len(compact_jev["jev_trace"])


def test_section_flags_and_logging_clarity(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "sections.json"
    dataset.write_text(json.dumps([
        {
            "question_id": "q-user-1", "question_type": "single-session-user",
            "question": "What is my job?", "answer": "engineer",
            "haystack_session_ids": ["s1"], "haystack_dates": ["2024-01-01"],
            "haystack_sessions": [[{"role": "user", "content": "I am an engineer."}]],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "q-temp-1", "question_type": "temporal-reasoning",
            "question": "When did I travel?", "answer": "June",
            "haystack_session_ids": ["s2"], "haystack_dates": ["2024-06-01"],
            "haystack_sessions": [[{"role": "user", "content": "I traveled in June."}]],
            "answer_session_ids": ["s2"],
        },
        {
            "question_id": "q-pref-1", "question_type": "single-session-preference",
            "question": "What drink do I like?", "answer": "tea",
            "haystack_session_ids": ["s3"], "haystack_dates": ["2024-02-01"],
            "haystack_sessions": [[{"role": "user", "content": "I like tea."}]],
            "answer_session_ids": ["s3"],
        },
    ]), encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setenv("RESULTS_DIR", str(results))

    # Test 1: --temporal flag filters only temporal-reasoning questions and logs section tag
    code = runner.main([
        "--mode", "baseline", "--dataset", str(dataset), "--retrieval-only",
        "--temporal", "--allow-model-fallback",
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "[q-temp-1 | temporal-reasoning] Starting baseline retrieval" in err
    assert "q-user-1" not in err
    assert "q-pref-1" not in err
    assert (results / "q-temp-1" / "baseline.json").exists()
    assert not (results / "q-user-1" / "baseline.json").exists()

    # Test 2: --preference flag filters only preference questions
    code = runner.main([
        "--mode", "baseline", "--dataset", str(dataset), "--retrieval-only",
        "--preference", "--allow-model-fallback",
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "[q-pref-1 | single-session-preference] Starting baseline retrieval" in err
    assert "q-user-1" not in err
    assert "q-temp-1" not in err
    assert (results / "q-pref-1" / "baseline.json").exists()

    # Test 3: Without flags, all cases are processed (default behavior preserved)
    code = runner.main([
        "--mode", "baseline", "--dataset", str(dataset), "--retrieval-only",
        "--allow-model-fallback",
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "[q-user-1 | single-session-user] Starting baseline retrieval" in err
    assert "[q-temp-1 | temporal-reasoning] Starting baseline retrieval" in err
    assert "[q-pref-1 | single-session-preference] Starting baseline retrieval" in err


