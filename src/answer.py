from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


ANSWER_INSTRUCTIONS = """Answer the question directly and concisely using the retrieved conversation evidence from past chats.
Track user preferences, entities, habits, gear, interests, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation or suggestion questions:
- The user is asking for personalized recommendations based on their ongoing interests, tastes, setup, or past activities from previous conversations.
- Even if the user asks for events, activities, or places 'around me' or in a city, do not say you don't know their location or give generic search tips. Instead, immediately ground your recommendations in their specific interests and languages from the conversation (e.g. if the user engages in language learning/exchange, specifically suggest cultural events where they can practice those languages, such as French and Spanish language exchange events, festivals, or conversation groups).
- If they ask for publications or conferences, identify their specific research domain (such as deep learning for medical imaging / AI in healthcare) and recommend conferences (e.g. MICCAI) and publications in that domain.
- If they ask for hotels, recommend hotel features matching their desired amenities (such as rooftop pools, balcony hot tubs, or skyline views).
- If they ask for accessories, recommend items compatible with their specific gear setup (such as Sony cameras).

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

