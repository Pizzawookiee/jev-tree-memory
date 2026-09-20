from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


ANSWER_INSTRUCTIONS = """Answer the question directly and concisely using the retrieved conversation evidence.
Track user preferences, entities, habits, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.
Only state that evidence is insufficient if the topic or activity is not mentioned in the evidence. Do not use outside knowledge."""


@dataclass
class ModelResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost: float


class OpenAIAnswerer:
    def __init__(self, api_key: str, model: str = "gpt-4o", session: requests.Session | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required unless --retrieval-only is used")
        self.api_key, self.model = api_key, model
        self.session = session or requests.Session()

    def call(self, question: str, evidence: str) -> ModelResult:
        start = time.perf_counter()
        response = self.session.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "instructions": ANSWER_INSTRUCTIONS,
                  "input": f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}", "temperature": 0}, timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("output_text") or "".join(
            part.get("text", "") for item in payload.get("output", []) for part in item.get("content", []) if part.get("type") == "output_text"
        )
        usage = payload.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cost = input_tokens / 1_000_000 * 2.50 + output_tokens / 1_000_000 * 10.00
        return ModelResult(text, input_tokens, output_tokens, (time.perf_counter() - start) * 1000, cost)

