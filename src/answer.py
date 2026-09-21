from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


ANSWER_INSTRUCTIONS = """Answer the question directly and concisely using the retrieved conversation evidence from past chats.
Track user preferences, entities, habits, equipment, background, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation, advice, or suggestion questions:
- Provide personalized recommendations that directly recall and utilize the user's specific ongoing interests, tastes, hobbies, equipment setup, or stated preferences from the dialogue.
- Ground the advice in the specific past conversation sessions that directly relate to the user's inquiry, ignoring unrelated topics in other sessions.
- Do not decline recommendations due to lack of real-time or location access; instead, recommend specific options, categories, or resources aligned with the user's remembered preferences and background.

Only state that evidence is insufficient if no relevant facts, user preferences, background, or topics related to the question are present in the evidence. Do not contradict the evidence."""


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

    def call(self, question: str, evidence: str, question_date: str | None = None) -> ModelResult:
        start = time.perf_counter()
        date_line = f"Current Interaction Date: {question_date}\n" if question_date else ""
        input_text = f"Retrieved conversation evidence:\n{evidence}\n\n{date_line}Current User Question to answer:\n{question}"
        response = self.session.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "instructions": ANSWER_INSTRUCTIONS,
                  "input": input_text, "temperature": 0}, timeout=120,
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

