@echo off
setlocal

REM Run from the repository root regardless of invocation directory.
cd /d "%~dp0"

if not exist "results" mkdir "results"

REM Use a fresh user-owned temp directory. This avoids permission errors from
REM stale pytest folders created by another Windows security context.
set "JEV_TEST_TEMP=%CD%\.test-tmp\run-%RANDOM%-%RANDOM%"
if not exist "%JEV_TEST_TEMP%" mkdir "%JEV_TEST_TEMP%"
if errorlevel 1 goto :error
set "TEMP=%JEV_TEST_TEMP%"
set "TMP=%JEV_TEST_TEMP%"

REM Pytest's cache is optional and can inherit inaccessible ACLs from an older
REM sandboxed run, so keep it disabled for this root smoke script.
set "PYTEST_ADDOPTS=-p no:cacheprovider"

REM ============================================================
REM ACTIVE COMMAND: local unit tests followed by one manual case
REM ============================================================

python -m pytest tests -q
if errorlevel 1 goto :error

python -m src.runner --mode jev-primary --trace-jev --limit 1 --resume

if errorlevel 1 goto :error

goto :success

REM ============================================================
REM TEST OPTIONS
REM Uncomment one command at a time to use it.
REM ============================================================

REM Run all unit tests:
REM python -m pytest tests -q

REM Run verbose unit tests:
REM python -m pytest tests -vv

REM Stop on the first test failure:
REM python -m pytest tests -q -x

REM Run one test file:
REM python -m pytest tests\test_retrieval.py -vv

REM ============================================================
REM RETRIEVAL-ONLY OPTIONS: no answer or judge calls
REM ============================================================

REM python -m src.runner --compare --limit 1 --retrieval-only
REM python -m src.runner --compare --limit 5 --retrieval-only
REM python -m src.runner --mode baseline --limit 5 --retrieval-only
REM python -m src.runner --mode jev-primary --limit 5 --retrieval-only
REM python -m src.runner --mode jev-primary --limit 1 --retrieval-only --trace-jev
REM python -m src.runner --mode jev-primary --limit 1 --retrieval-only --trace-jev --full-jev-diagnostics
REM python -m src.runner --mode jev-primary --case-id CASE_ID --retrieval-only --trace-jev

REM ============================================================
REM MANUAL-JUDGE OPTIONS: export prompts without judge API calls
REM ============================================================

REM Export evidence-only prompts for ChatGPT to fill in suggested_answer:
REM python -m src.runner --compare --limit 1 --manual-answer
REM Import those answers and generate the manual judge prompts:
REM python -m src.runner --compare --limit 1 --manual-judge --manual-answer-input results\manual_answer_prompts.jsonl --resume
REM python -m src.runner --compare --limit 1 --manual-judge
REM python -m src.runner --compare --limit 5 --manual-judge
REM python -m src.runner --compare --limit 5 --manual-judge --manual-judge-output results\custom_manual_prompts.jsonl
REM python -m src.runner --compare --limit 5 --manual-judge --reuse-generations
REM python -m src.runner --judge-only --manual-judge --resume
REM python -m src.runner --compare --all --manual-judge --reuse-generations --resume

REM ============================================================
REM ANSWER-GENERATION OPTIONS
REM ============================================================

REM python -m src.runner --compare --limit 5 --answer-only
REM python -m src.runner --compare --limit 5 --answer-only --eval-model gpt-4o-mini
REM python -m src.runner --compare --limit 5 --answer-only --eval-model gpt-4o
REM python -m src.runner --compare --limit 5 --answer-only --reuse-generations

REM ============================================================
REM OFFICIAL LONGMEMEVAL EXPORT AND EVALUATION OPTIONS
REM ============================================================

REM Normal runs export official JSONL and evaluate with gpt-4o-2024-08-06:
REM python -m src.runner --mode jev-primary --limit 5 --resume
REM Export answers in official format without running the judge:
REM python -m src.runner --mode jev-primary --all --answer-only --resume
REM Judge cached answers with the official evaluator:
REM python -m src.runner --mode jev-primary --all --judge-only --resume

REM ============================================================
REM FULL BENCHMARK OPTIONS
REM ============================================================

REM python -m src.runner --mode baseline --all --eval-model gpt-4o --resume
REM python -m src.runner --mode jev-primary --all --eval-model gpt-4o --resume
REM python -m src.runner --compare --all --eval-model gpt-4o --resume

:success
echo.
echo Tests and benchmark smoke test completed successfully.
echo.
pause
exit /b 0

:error
set "exit_code=%errorlevel%"
echo.
echo A command failed with exit code %exit_code%.
echo.
pause
exit /b %exit_code%
