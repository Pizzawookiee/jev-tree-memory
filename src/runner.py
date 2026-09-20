from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from .answer import OpenAIAnswerer
from .benchmark import load_cases
from .config import Settings, USE_DIRECT_CONTEXT
from .context import expand_and_pack
from .database import MemoryDB
from .embeddings import LocalEmbedder
from .ingest import ingest_case
from .jev import JEV_INPUT_USD_PER_MILLION_TOKENS, JevClient, JevMemoryRouter
from .jev_cache import JevDecisionCache
from .manual_answer import export_manual_answer_prompts, load_manual_answers, retrieval_hash
from .manual_judge import export_manual_prompts
from .metrics import jev_metrics, retrieval_metrics
from .official_eval import OfficialLongMemEvalEvaluator, export_hypotheses
from .rerank import CrossEncoderReranker
from .retrieve import Retriever
from .search_policy import JevSearchPolicy
from .tree import MemoryTree

assert USE_DIRECT_CONTEXT is False


def _artifact_path(settings: Settings, case_id: str, mode: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)
    return settings.results_dir / safe / f"{mode}.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load_artifact(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _query_type_from_benchmark(value: str) -> str:
    mapping = {"single-session-preference": "PREFERENCE", "temporal-reasoning": "TEMPORAL",
               "knowledge-update": "LATEST_STATE", "multi-session": "MULTI_SESSION"}
    return mapping.get(value, "POINT_LOOKUP")


def _retrieve_case(case, mode: str, settings: Settings, args) -> dict:
    print(f"[{case.case_id}] Starting {mode} retrieval", file=sys.stderr, flush=True)
    stage_timings: dict[str, float] = {}
    case_dir = settings.results_dir / "databases"
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / f"{case.case_id}.sqlite"
    db = MemoryDB(db_path)
    embedding_identity = json.dumps(
        {
            "model": settings.embedding_model,
            "backend": settings.embedding_backend,
            "onnx_file": settings.embedding_onnx_file,
        },
        sort_keys=True,
    )
    if (
        db.conn.execute("SELECT 1 FROM turns LIMIT 1").fetchone()
        and db.metadata("embedding_identity") != embedding_identity
    ):
        print(
            f"[{case.case_id}] Embedding configuration changed; rebuilding the case database",
            file=sys.stderr,
            flush=True,
        )
        db.close()
        db_path.unlink()
        db = MemoryDB(db_path)
    print(f"[{case.case_id}] Loading embedding model...", file=sys.stderr, flush=True)
    stage_started = time.perf_counter()
    embedder = LocalEmbedder(
        settings.embedding_model,
        args.allow_model_fallback,
        settings.embedding_batch_size,
        settings.embedding_backend,
        settings.embedding_onnx_file,
        settings.embedding_threads,
        settings.embedding_query_prompt_name,
    )
    stage_timings["embedding_model_load_ms"] = (time.perf_counter() - stage_started) * 1000
    print(
        f"[{case.case_id}] Embedding model ready in {stage_timings['embedding_model_load_ms'] / 1000:.2f}s",
        file=sys.stderr, flush=True,
    )
    print(f"[{case.case_id}] Loading cross-encoder reranker...", file=sys.stderr, flush=True)
    stage_started = time.perf_counter()
    reranker = CrossEncoderReranker(
        settings.reranker_model, args.allow_model_fallback, settings.reranker_batch_size
    )
    stage_timings["reranker_model_load_ms"] = (time.perf_counter() - stage_started) * 1000
    print(
        f"[{case.case_id}] Cross-encoder ready in {stage_timings['reranker_model_load_ms'] / 1000:.2f}s",
        file=sys.stderr, flush=True,
    )
    if not db.conn.execute("SELECT 1 FROM turns LIMIT 1").fetchone():
        print(
            f"[{case.case_id}] Splitting and embedding {len(case.turns):,} raw turns...",
            file=sys.stderr,
            flush=True,
        )
        stage_started = time.perf_counter()
        sentence_ids = ingest_case(db, case, embedder)
        db.set_metadata("embedding_identity", embedding_identity)
        stage_timings["ingestion_ms"] = (time.perf_counter() - stage_started) * 1000
        print(
            f"[{case.case_id}] Ingestion complete: {len(sentence_ids):,} sentences indexed "
            f"in {stage_timings['ingestion_ms'] / 1000:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    else:
        sentence_count = db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0]
        print(
            f"[{case.case_id}] Using cached ingestion: {sentence_count:,} sentences",
            file=sys.stderr,
            flush=True,
        )
    print(f"[{case.case_id}] Initializing memory tree...", file=sys.stderr, flush=True)
    stage_started = time.perf_counter()
    tree = MemoryTree(db, embedder, settings.max_depth)
    tree.initialize()
    stage_timings["tree_initialization_ms"] = (time.perf_counter() - stage_started) * 1000
    print(
        f"[{case.case_id}] Memory tree ready in {stage_timings['tree_initialization_ms'] / 1000:.2f}s",
        file=sys.stderr, flush=True,
    )
    retriever = Retriever(db, embedder, tree)
    plan = None
    jev_client = None
    jev_router = None
    routing_profile = None
    reranked = None
    packed_context = None
    context_turn_ids = None
    if mode == "baseline":
        candidates = retriever.baseline(case.question)
        query_type = _query_type_from_benchmark(case.question_type)
    else:
        jev_client = JevClient(settings.typesafe_api_key, settings.jev_model, args.allow_model_fallback)
        routing_profile = "full" if args.full_jev_diagnostics else "lean"
        prior = db.conn.execute(
            "SELECT selected_action FROM jev_decisions WHERE phase='memory-routing' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prior_profile = None
        if prior:
            try:
                prior_profile = json.loads(prior["selected_action"]).get("routing_profile")
            except (TypeError, ValueError):
                prior_profile = None
        has_existing_routes = db.conn.execute("SELECT 1 FROM node_evidence LIMIT 1").fetchone() is not None
        if has_existing_routes and prior_profile != routing_profile:
            print(
                f"Jev routing profile changed from {prior_profile or 'legacy'} to {routing_profile}; rebuilding routes",
                file=sys.stderr, flush=True,
            )
            db.conn.execute("DELETE FROM node_evidence")
            db.conn.execute("DELETE FROM jev_decisions WHERE phase='memory-routing'")
            db.conn.execute(
                "UPDATE sentences SET memory_type=NULL,retrieval_priority=NULL,temporal_scope=NULL"
            )
            db.conn.commit()
        sentence_count = int(db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0])
        routed_count = int(db.conn.execute("SELECT count(DISTINCT sentence_id) FROM node_evidence").fetchone()[0])
        if routed_count < sentence_count:
            batch_size = args.jev_routing_batch_size or settings.jev_routing_batch_size
            print(
                f"Jev routing profile: {routing_profile} | batch size: {batch_size:,} turns per request",
                file=sys.stderr, flush=True,
            )
            decision_cache = JevDecisionCache(settings.results_dir / "jev_routing_cache.sqlite")
            jev_router = JevMemoryRouter(
                db, tree, jev_client, profile=routing_profile, cache=decision_cache
            )
            try:
                jev_router.route_all(case.case_id, batch_size=batch_size)
            finally:
                decision_cache.close()
        else:
            print(f"Jev routing: using cached routes for {routed_count:,} sentences", file=sys.stderr)
        policy = JevSearchPolicy(db, tree, jev_client, settings.max_search_steps)
        print("Jev search: classifying question and selecting branches...", file=sys.stderr)
        plan = policy.plan(case.case_id, case.question)
        if args.trace_jev:
            print(f"Jev query decision: {json.dumps(plan.steps[-1], sort_keys=True)}", file=sys.stderr)
        candidates = retriever.jev(case.question, plan)
        reranked = reranker.rerank(case.question, candidates, limit=20)
        packed_context, context_turn_ids = expand_and_pack(db, reranked, plan.query_type)
        print("Jev search: checking actual answerer-context sufficiency...", file=sys.stderr)
        plan = policy.sufficiency(case.case_id, case.question, packed_context, plan)
        if args.trace_jev:
            print(f"Jev sufficiency decision: {json.dumps(plan.steps[-1], sort_keys=True)}", file=sys.stderr)
        if plan.steps[-1]["action"] == "GLOBAL_RESCUE":
            print("Jev search: context insufficient; running full-corpus global rescue...", file=sys.stderr)
            candidates = retriever.jev(case.question, plan)
            reranked = reranker.rerank(case.question, candidates, limit=20)
            packed_context, context_turn_ids = expand_and_pack(db, reranked, plan.query_type)
        total_tokens = jev_client.input_tokens + jev_client.output_tokens
        average_latency = jev_client.total_latency_ms / jev_client.request_count if jev_client.request_count else 0.0
        print(
            f"Jev complete: {jev_client.request_count:,} requests this run | "
            f"{total_tokens:,} tokens | estimated cost: ${jev_client.estimated_cost:.6f} | "
            f"average latency: {average_latency:,.0f} ms",
            file=sys.stderr,
        )
        query_type = plan.query_type
    if reranked is None:
        reranked = reranker.rerank(case.question, candidates, limit=20)
    if packed_context is None or context_turn_ids is None:
        packed_context, context_turn_ids = expand_and_pack(db, reranked, query_type)
    metrics = retrieval_metrics(case, reranked, context_turn_ids, db)
    if plan:
        metrics["jev"] = jev_metrics(case, reranked, plan, db)
        metrics["jev"].update({
            "routing_profile": routing_profile,
            "cross_case_cache_hits": jev_router.cache_hits if jev_router else 0,
            "trivial_turns_filtered": jev_router.trivial_turns if jev_router else 0,
        })
        corpus_size = max(1, db.conn.execute("SELECT count(*) FROM sentences").fetchone()[0])
        metrics["candidate_reduction_vs_corpus"] = 1 - len(candidates) / corpus_size
    artifact = {
        "case_id": case.case_id, "mode": mode, "status": "retrieval-complete",
        "question": case.question, "query_type": query_type,
        "backends": {"embeddings": embedder.backend, "reranker": reranker.backend,
                     "jev": jev_client.backend if jev_client else None, "answerer": None},
        "retrieved_evidence": [item.to_dict() for item in reranked], "context_turn_ids": context_turn_ids,
        "packed_context": packed_context, "retrieval_metrics": metrics, "stage_timings_ms": stage_timings,
    }
    if mode == "jev-primary":
        artifact["jev_usage"] = {
            "routing_profile": routing_profile,
            "requests": jev_client.request_count,
            "input_tokens": jev_client.input_tokens,
            "output_tokens": jev_client.output_tokens,
            "input_usd_per_million_tokens": JEV_INPUT_USD_PER_MILLION_TOKENS,
            "output_usd_per_million_tokens": 0.0,
            "estimated_cost": jev_client.estimated_cost,
            "cross_case_cache_hits": jev_router.cache_hits if jev_router else 0,
            "trivial_turns_filtered": jev_router.trivial_turns if jev_router else 0,
        }
        artifact["jev_trace"] = [dict(row) for row in db.conn.execute(
            "SELECT * FROM jev_decisions WHERE case_id=? ORDER BY id", (case.case_id,))]
    db.close()
    return artifact


def _generate_answer(case, artifact: dict, settings: Settings, args) -> None:
    if args.reuse_generations and artifact.get("answer"):
        return
    if args.allow_model_fallback:
        excerpts = [item["content"] for item in artifact.get("retrieved_evidence", [])[:3]]
        artifact.update({
            "answer": " ".join(excerpts) if excerpts else "Insufficient retrieved evidence.",
            "answer_token_usage": {"input": 0, "output": 0},
            "answer_estimated_cost": 0.0,
            "answer_latency_ms": 0.0,
            "answer_api_called": False,
        })
        artifact["backends"]["answerer"] = "deterministic-extractive-fallback"
        return
    answer = OpenAIAnswerer(settings.openai_api_key or "", args.eval_model).call(
        case.question, artifact["packed_context"])
    artifact.update({
        "answer": answer.text,
        "answer_token_usage": {"input": answer.input_tokens, "output": answer.output_tokens},
        "answer_estimated_cost": answer.estimated_cost,
        "answer_latency_ms": answer.latency_ms,
        "answer_api_called": True,
    })
    artifact["backends"]["answerer"] = args.eval_model


def _apply_manual_answer(case, artifact: dict, answers: dict[str, dict]) -> None:
    answer_id = f"{case.case_id}:{artifact['mode']}"
    entry = answers.get(answer_id)
    if not entry:
        raise ValueError(f"Manual answer input has no entry for {answer_id}")
    expected_hash = retrieval_hash(case.question, artifact.get("packed_context", ""))
    if entry.get("retrieval_hash") != expected_hash:
        raise ValueError(
            f"Manual answer retrieval hash does not match current evidence for {answer_id}; "
            "regenerate the manual answer export"
        )
    answer = entry.get("suggested_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Manual answer input has no suggested_answer for {answer_id}")
    artifact.update({
        "answer": answer.strip(),
        "answer_token_usage": {"input": 0, "output": 0},
        "answer_estimated_cost": 0.0,
        "answer_latency_ms": 0.0,
        "answer_api_called": False,
    })
    artifact.pop("judge", None)
    artifact["backends"]["answerer"] = "manual-chatgpt-import"


def run_case(case, modes: list[str], settings: Settings, args) -> list[dict]:
    outputs = []
    db_path = settings.results_dir / "databases" / f"{case.case_id}.sqlite"
    if db_path.exists() and not args.resume and not args.judge_only and not args.reuse_generations:
        db_path.unlink()
    for mode in modes:
        started = time.perf_counter()
        path = _artifact_path(settings, case.case_id, mode)
        cached = _load_artifact(path)
        if args.judge_only:
            if not cached or (not cached.get("answer") and not args.manual_answer_input):
                raise RuntimeError(f"--judge-only requires a cached answer at {path}")
            artifact = cached
        elif args.reuse_generations and cached and cached.get("answer"):
            artifact = cached
        elif args.resume and cached and (
            args.retrieval_only or (args.answer_only and cached.get("answer")) or
            (args.manual_answer and cached.get("packed_context")) or
            (args.manual_judge and (cached.get("answer") or (args.manual_answer_input and cached.get("packed_context")))) or
            (not args.answer_only and not args.manual_judge and cached.get("answer"))
        ):
            artifact = cached
        else:
            artifact = _retrieve_case(case, mode, settings, args)

        if args.retrieval_only:
            artifact["status"] = "retrieval-complete"
        elif args.manual_answer:
            artifact["status"] = "awaiting_manual_answer"
        else:
            if args.manual_answer_input:
                _apply_manual_answer(case, artifact, args.manual_answers)
            elif not args.judge_only and not artifact.get("answer"):
                _generate_answer(case, artifact, settings, args)
            if args.manual_judge:
                automatic_judge_enabled = False
                openai_batch_judge_enabled = False
                assert automatic_judge_enabled is False
                assert openai_batch_judge_enabled is False
                artifact["status"] = "awaiting_manual_judgment"
            elif args.answer_only:
                artifact["status"] = "answer-complete"
            else:
                # LongMemEval's authoritative judge consumes the completed
                # hypotheses as a separate dataset-level evaluation pass.
                artifact["status"] = "answer-complete"
        artifact["total_latency_ms"] = artifact.get("total_latency_ms", 0) + (time.perf_counter() - started) * 1000
        _write_json(path, artifact)
        outputs.append(artifact)
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mode", choices=["baseline", "jev-primary"])
    group.add_argument("--compare", action="store_true")
    parser.add_argument("--case-id")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--retrieval-only", action="store_true")
    phase.add_argument("--answer-only", action="store_true")
    phase.add_argument("--judge-only", action="store_true")
    phase.add_argument("--manual-answer", action="store_true")
    parser.add_argument("--manual-judge", action="store_true")
    parser.add_argument("--manual-judge-output", type=Path)
    parser.add_argument("--manual-answer-output", type=Path)
    parser.add_argument("--manual-answer-input", type=Path)
    parser.add_argument(
        "--hypothesis-output", type=Path,
        help="official LongMemEval JSONL path (single-mode runs only)",
    )
    parser.add_argument(
        "--official-reference", type=Path,
        help="reference JSON; defaults to LONGMEMEVAL_ORACLE_PATH, a sibling oracle, or the dataset",
    )
    parser.add_argument("--eval-model")
    parser.add_argument("--reuse-generations", action="store_true")
    parser.add_argument("--trace-jev", action="store_true")
    parser.add_argument("--jev-routing-batch-size", type=int)
    parser.add_argument(
        "--full-jev-diagnostics", action="store_true",
        help="use full five-question Jev metadata routing and disable trivial-turn filtering",
    )
    parser.add_argument("--allow-model-fallback", action="store_true")
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args(argv)
    if args.manual_judge and args.retrieval_only:
        parser.error("--manual-judge cannot be combined with --retrieval-only")
    if args.manual_judge and args.manual_answer:
        parser.error("--manual-judge cannot be combined with --manual-answer")
    if args.case_id and args.all:
        parser.error("--case-id cannot be combined with --all")
    if not args.mode and not args.compare and not args.judge_only:
        parser.error("one of --mode or --compare is required")
    if args.manual_judge_output and not args.manual_judge:
        parser.error("--manual-judge-output requires --manual-judge")
    if args.manual_answer_output and not args.manual_answer:
        parser.error("--manual-answer-output requires --manual-answer")
    if args.manual_answer_input and not args.manual_judge:
        parser.error("--manual-answer-input requires --manual-judge")
    if args.hypothesis_output and args.compare:
        parser.error("--hypothesis-output cannot be combined with --compare")
    if args.jev_routing_batch_size is not None and args.jev_routing_batch_size < 1:
        parser.error("--jev-routing-batch-size must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    args.eval_model = args.eval_model or settings.openai_model
    args.manual_answers = load_manual_answers(args.manual_answer_input) if args.manual_answer_input else {}
    dataset = args.dataset or settings.longmemeval_path
    if not dataset or not dataset.exists():
        print("LONGMEMEVAL_PATH or --dataset must point to a LongMemEval JSON/JSONL file", file=sys.stderr)
        return 2
    modes = ["baseline", "jev-primary"] if args.compare or not args.mode else [args.mode]
    cases = list(load_cases(dataset, args.case_id, None if args.all else args.limit))
    if not cases:
        print("No matching LongMemEval cases", file=sys.stderr)
        return 2
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        print("The selected LongMemEval cases contain duplicate question_id values", file=sys.stderr)
        return 2
    if args.all and len(cases) != 500:
        print(
            f"--all requires the official 500-case LongMemEval set; loaded {len(cases):,} cases",
            file=sys.stderr,
        )
        return 2

    run_official_evaluation = (
        not args.retrieval_only
        and not args.answer_only
        and not args.manual_answer
        and not args.manual_judge
    )
    official_reference = None
    if run_official_evaluation:
        official_reference = args.official_reference or settings.longmemeval_oracle_path
        if official_reference is None:
            sibling_oracle = dataset.with_name("longmemeval_oracle.json")
            official_reference = sibling_oracle if sibling_oracle.exists() else dataset
        if not official_reference.exists():
            print(
                f"Official LongMemEval reference file does not exist: {official_reference}",
                file=sys.stderr,
            )
            return 2
    records: list[tuple[object, dict]] = []
    try:
        for case in cases:
            records.extend((case, artifact) for artifact in run_case(case, modes, settings, args))
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    hypothesis_exports: dict[str, Path] = {}
    if not args.retrieval_only and not args.manual_answer:
        try:
            for mode in modes:
                output = args.hypothesis_output or (
                    settings.results_dir / f"longmemeval_{mode}_hypotheses.jsonl"
                )
                output, count = export_hypotheses(records, output, mode)
                hypothesis_exports[mode] = output
                print(f"Official LongMemEval hypotheses ({mode}): {output} ({count:,} answers)")
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.manual_answer:
        output = args.manual_answer_output or settings.results_dir / "manual_answer_prompts.jsonl"
        jsonl_path, markdown_path, count = export_manual_answer_prompts(records, output, args.eval_model)
        print(f"Generated {count} manual answer prompts.")
        print("No answer or judge API calls were made.")
        print(f"\nJSONL: {jsonl_path}\nMarkdown: {markdown_path}")
    elif args.manual_judge:
        output = args.manual_judge_output or settings.results_dir / "manual_judge_prompts.jsonl"
        jsonl_path, markdown_path, count = export_manual_prompts(records, output, args.eval_model)
        print(f"Generated {count} manual judge prompts.")
        print("No judge API calls were made.")
        if any(artifact.get("answer_api_called") for _, artifact in records):
            print("Answer-generation API calls were made for uncached answers.")
        else:
            print("No answer-generation API calls were made.")
        print(f"\nJSONL: {jsonl_path}\nMarkdown: {markdown_path}")
    elif run_official_evaluation:
        assert official_reference is not None
        try:
            evaluator = OfficialLongMemEvalEvaluator(settings.openai_api_key or "")
            for mode, hypothesis_path in hypothesis_exports.items():
                result_path, metrics_path, metrics = evaluator.evaluate(
                    hypothesis_path, official_reference
                )
                print(
                    f"Official LongMemEval evaluation ({mode}): "
                    f"overall={metrics['overall_accuracy']:.4f}, "
                    f"task-averaged={metrics['task_averaged_accuracy']:.4f}"
                )
                print(f"Evaluation log: {result_path}")
                print(f"Metrics: {metrics_path}")
        except (RuntimeError, ValueError, requests.RequestException) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        print(json.dumps([{"case_id": artifact["case_id"], "mode": artifact["mode"],
                           "status": artifact["status"], "metrics": artifact["retrieval_metrics"]}
                          for _, artifact in records], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
