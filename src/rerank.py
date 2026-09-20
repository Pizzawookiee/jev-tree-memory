from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from .models import Evidence


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str, allow_fallback: bool = False, batch_size: int = 64):
        self.batch_size = batch_size
        if allow_fallback:
            self.model = None
            self.backend = "lexical-reranker-fallback"
            self._fallback = True
            return
        try:
            self.model = _load_cross_encoder(model_name)
            self.backend = model_name
            self._fallback = False
        except Exception:
            raise

    def rerank(self, question: str, evidence: list[Evidence], limit: int = 20) -> list[Evidence]:
        if not evidence:
            return []
        if self._fallback:
            query = set(re.findall(r"[a-z0-9]+", question.lower()))
            scores = [len(query & set(re.findall(r"[a-z0-9]+", item.content.lower()))) / max(len(query), 1) for item in evidence]
        else:
            raw = np.asarray(
                self.model.predict(
                    [(question, item.content) for item in evidence],
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ),
                dtype=float,
            )
            scores = (1.0 / (1.0 + np.exp(-raw))).tolist()
        for item, score in zip(evidence, scores):
            item.cross_score = float(score)
            item.final_score = 0.60 * item.hybrid_score + 0.40 * item.cross_score
        return sorted(evidence, key=lambda item: item.final_score, reverse=True)[:limit]
