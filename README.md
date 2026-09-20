# Jev-centered agentic memory MVP

This repository runs a controlled LongMemEval comparison between global hybrid
retrieval (`baseline`) and Jev-directed n-ary tree retrieval (`jev-primary`). Raw
turns and sentences are immutable and globally indexed in both modes. Jev only
controls metadata, tree routing, candidate allocation, and search expansion.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY`, `OPENAI_API_KEY`, and
`LONGMEMEVAL_PATH`. Set `LONGMEMEVAL_ORACLE_PATH` to the official
`longmemeval_oracle.json` for the canonical evaluation command. If it is not
set, the runner uses a sibling oracle file when available, then falls back to
the selected dataset (the evaluator only reads question IDs, types, questions,
and reference answers). The local `.env` is ignored by Git. Variables supplied
by the shell take precedence over values in the file.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` and
`cross-encoder/ms-marco-MiniLM-L-6-v2` into the local Hugging Face cache.
`EMBEDDING_MODEL` and `RERANKER_MODEL` may instead point at pre-cached local
model directories (useful on hosts whose outbound TLS proxy is not trusted by
Python).

For Snowflake Arctic XS with CPU INT8 ONNX inference, install the project again
after updating so the ONNX dependencies are present, then use:

```powershell
python -m pip install -e ".[dev]"
```

```env
EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-xs
EMBEDDING_BACKEND=onnx
EMBEDDING_ONNX_FILE=onnx/model_int8.onnx
EMBEDDING_QUERY_PROMPT_NAME=query
EMBEDDING_THREADS=4
EMBEDDING_BATCH_SIZE=128
```

The runner sends benchmark questions through Snowflake's `query` prompt and
encodes stored evidence without a prompt, as required by the model. ONNX runs
through `CPUExecutionProvider`. Changing the model, backend, or ONNX graph
automatically invalidates and rebuilds incompatible per-case databases.

```powershell
python -m src.runner --compare --case-id <question_id>
python -m src.runner --compare --limit 5
python -m src.runner --mode jev-primary --all --resume
python -m src.runner --mode jev-primary --case-id <question_id> --retrieval-only --trace-jev
python -m src.runner --compare --limit 5 --manual-judge
```

A normal answer run now uses the official LongMemEval evaluation protocol. For
each mode it writes an exact two-field submission file:

```text
results/longmemeval_jev-primary_hypotheses.jsonl
results/longmemeval_baseline_hypotheses.jsonl
```

Each line contains only `question_id` and `hypothesis`. The runner then judges
it with the benchmark-pinned `gpt-4o-2024-08-06`, temperature 0, and the
category-specific official prompts. It writes the upstream-compatible
`.eval-results-gpt-4o` JSONL log and a `.metrics.json` sidecar with overall,
task-averaged, per-type, and abstention accuracy plus judge usage.

Generate a submission without paying for the judge with:

```powershell
python -m src.runner --mode jev-primary --all --answer-only --resume
```

Evaluate already-generated answers without rerunning retrieval or answering:

```powershell
python -m src.runner --mode jev-primary --all --judge-only --resume
```

Use `--hypothesis-output PATH` to override the JSONL path for a single mode and
`--official-reference PATH` to override the evaluation reference file.

Results and per-case SQLite files are written to `results/`. `--resume` skips a
mode only when its completed JSON artifact already exists. For offline smoke
tests only, `--allow-model-fallback` enables deterministic hashing embeddings,
lexical reranking, deterministic Jev routing, and an extractive smoke-test answer. Fallback artifacts identify
their backend and must not be reported as real Jev/GPT-4o benchmark runs.

Jev-primary runs display live sentence-routing progress, request and token
counts, elapsed time, and estimated time remaining. `--trace-jev` additionally
prints the query-routing and evidence-sufficiency actions. If sentence routes
already exist in the case database, the runner reports that it is reusing them.
Lean Jev routing is the default. Jev decides the branch and retrieval priority;
memory type, temporal scope, and topic creation use deterministic local rules.
Exact, conservative assistant acknowledgements are routed locally but remain in
the global indexes. Substantive routing decisions are cached by model, profile,
role, content hash, and question-schema hash in `results/jev_routing_cache.sqlite`
and reused across cases. Cache hits still create inspectable case decision logs.

Routing is performed once per raw turn and applied to that turn's sentences,
then batched into 25 turns per Jev request by default. Each turn appears once
in the shared structured request state rather than being repeated per question. Set
`JEV_ROUTING_BATCH_SIZE` in `.env` or pass `--jev-routing-batch-size N` to tune
it. For example, 550 turns require about 22 routing requests at the default
size instead of 5,700 sentence requests; query routing and sufficiency add two.
The result artifact records Jev input/output tokens and estimated input cost at
the current Jev price configured in the code.

For diagnostic runs that require Jev to decide all five metadata fields and to
process even trivial acknowledgements, use:

```powershell
python -m src.runner --mode jev-primary --limit 1 --full-jev-diagnostics --trace-jev
```

Full and lean decisions occupy separate cache namespaces. Deterministic fallback
decisions are also isolated from live Jev cache entries.

## Manual judging

`--manual-judge` generates answers and exports the exact category-specific
LongMemEval evaluator prompts without calling the judge API. It writes
`results/manual_judge_prompts.jsonl` and a copy-friendly Markdown sibling.
Existing labels survive regeneration while the answer hash is unchanged.

To generate both answers and judgments manually, first export answer prompts:

```powershell
python -m src.runner --compare --limit 1 --manual-answer
```

Send `results\manual_answer_prompts.jsonl` to ChatGPT and ask it to fill each
`suggested_answer` using only that entry's prompts, without changing any other
field. Save the completed JSONL, then import it while creating judge prompts:

```powershell
python -m src.runner --compare --limit 1 --manual-judge --manual-answer-input results\manual_answer_prompts.jsonl --resume
```

This path makes no OpenAI API answer or judge calls. Retrieval hashes prevent
answers generated from stale evidence from being silently imported. The export
never contains LongMemEval reference answers; those appear only in the later
judge prompts.

After filling `manual_response` (`yes`/`no` or `CORRECT`/`WRONG`) and
`manual_label` (`1`/`0`), aggregate only completed labels with:

```powershell
python -m src.metrics --manual-labels results\manual_judge_prompts.jsonl
```

Run the unit suite and one comparison/export smoke case from any directory with:

```powershell
run_tests.bat
```
