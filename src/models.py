from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MemoryType = Literal["fact", "preference", "event", "procedure", "state", "other"]
Priority = Literal["low", "normal", "high"]
TemporalScope = Literal["timeless", "historical", "current", "unknown"]
QueryType = Literal["POINT_LOOKUP", "PREFERENCE", "LATEST_STATE", "TEMPORAL", "MULTI_SESSION", "AGGREGATION"]


@dataclass(frozen=True)
class TurnInput:
    session_id: str
    session_index: int
    turn_index: int
    role: str
    content: str
    session_time: str | None


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question_type: str
    question: str
    question_date: str | None
    turns: tuple[TurnInput, ...]
    reference_answer: Any
    answer_session_ids: tuple[str, ...]
    answer_turn_keys: tuple[tuple[str, int], ...]


@dataclass
class MemoryDecision:
    memory_type: MemoryType
    retrieval_priority: Priority
    temporal_scope: TemporalScope
    selected_node_ids: list[str]
    create_topic: bool
    proposed_topic_label: str | None
    confidence: float
    probabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPlan:
    query_type: QueryType
    selected_node_ids: list[str]
    branch_probabilities: dict[str, float]
    global_rescue: bool
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Evidence:
    sentence_id: int
    turn_id: int
    session_id: str
    session_time: str | None
    role: str
    content: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    jev_score: float = 0.0
    priority_score: float = 0.5
    hybrid_score: float = 0.0
    cross_score: float = 0.0
    final_score: float = 0.0
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

