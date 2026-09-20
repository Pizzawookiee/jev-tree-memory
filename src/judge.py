from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .answer import ModelResult
from .judge_prompt import JudgePrompt, build_judge_prompt


def judge_request_payload(prompt: JudgePrompt, model: str) -> dict:
    messages = []
    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})
    messages.append({"role": "user", "content": prompt.user_prompt})
    return {"model": model, "messages": messages, "n": 1, "temperature": 0, "max_tokens": 10}


def batch_request(prompt: JudgePrompt, model: str, custom_id: str) -> dict:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions",
            "body": judge_request_payload(prompt, model)}


class OpenAIJudge:
    def __init__(self, api_key: str, model: str = "gpt-4o", session: requests.Session | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for automatic judging")
        self.api_key, self.model = api_key, model
        self.session = session or requests.Session()

    def prompt(self, question_id: str, question_type: str, question: str, reference: object, hypothesis: str) -> JudgePrompt:
        return build_judge_prompt(question_id, question, str(reference), hypothesis, question_type=question_type)

    def request(self, prompt: JudgePrompt) -> dict:
        return judge_request_payload(prompt, self.model)

    def judge(self, question_id: str, question_type: str, question: str, reference: object,
              hypothesis: str) -> tuple[dict, ModelResult]:
        prompt = self.prompt(question_id, question_type, question, reference, hypothesis)
        start = time.perf_counter()
        response = self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=self.request(prompt), timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        correct = "yes" in text.lower()
        usage = payload.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        cost = input_tokens / 1_000_000 * 2.50 + output_tokens / 1_000_000 * 10.00
        return {"correct": correct, "explanation": text}, ModelResult(
            text, input_tokens, output_tokens, (time.perf_counter() - start) * 1000, cost)


class OpenAIBatchJudge:
    def __init__(self, api_key: str, model: str = "gpt-4o", session: requests.Session | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for batch judging")
        self.api_key, self.model = api_key, model
        self.session = session or requests.Session()

    def request(self, prompt: JudgePrompt, custom_id: str) -> dict:
        return batch_request(prompt, self.model, custom_id)

    def submit(self, requests_: list[dict], output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "judge_batch_input.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests_), encoding="utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with path.open("rb") as handle:
            upload = self.session.post("https://api.openai.com/v1/files", headers=headers,
                                       data={"purpose": "batch"}, files={"file": handle}, timeout=120)
        upload.raise_for_status()
        batch = self.session.post(
            "https://api.openai.com/v1/batches", headers={**headers, "Content-Type": "application/json"},
            json={"input_file_id": upload.json()["id"], "endpoint": "/v1/chat/completions",
                  "completion_window": "24h", "metadata": {"description": "LongMemEval judge"}}, timeout=120,
        )
        batch.raise_for_status()
        return {"input_path": str(path), **batch.json()}

