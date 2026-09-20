from __future__ import annotations

from collections import defaultdict

from .database import MemoryDB
from .models import Evidence

BUDGETS = {"POINT_LOOKUP": 1500, "PREFERENCE": 3500, "LATEST_STATE": 2500, "TEMPORAL": 5000,
           "MULTI_SESSION": 7500, "AGGREGATION": 5000}


def expand_and_pack(db: MemoryDB, evidence: list[Evidence], query_type: str, maximum_budget: int = 7500) -> tuple[str, list[int]]:
    budget = min(BUDGETS.get(query_type, 5000), maximum_budget)
    selected_turns: dict[int, object] = {}
    selected_sessions: set[tuple[int, str, str | None]] = set()
    session_rank: dict[tuple[int, str, str | None], int] = {}
    used = 0

    # Spend the budget in retrieval-rank order. Adjacent turns are considered
    # only after the directly retrieved turn, so chronology cannot crowd the
    # strongest evidence out of the final context.
    for rank, item in enumerate(evidence):
        row = db.conn.execute("SELECT * FROM turns WHERE id=?", (item.turn_id,)).fetchone()
        if not row:
            continue
        session_key = (int(row["session_index"]), row["session_id"], row["session_time"])
        candidates = [row]
        candidates.extend(db.conn.execute(
            "SELECT * FROM turns WHERE case_id=? AND session_index=? AND turn_index BETWEEN ? AND ?",
            (row["case_id"], row["session_index"], max(0, row["turn_index"] - 1), row["turn_index"] + 1),
        ))
        # The hit itself is first; adjacent context follows in chronological order.
        candidates = [row] + sorted(
            (candidate for candidate in candidates[1:] if int(candidate["id"]) != int(row["id"])),
            key=lambda value: value["turn_index"],
        )
        for candidate in candidates:
            turn_id = int(candidate["id"])
            if turn_id in selected_turns:
                continue
            header_cost = 0
            if session_key not in selected_sessions:
                header = f"\n[Session {row['session_id']} | {row['session_time'] or 'date unknown'}]\n"
                header_cost = max(1, len(header) // 4)
            line = f"{candidate['role']}: {candidate['content']}\n"
            line_cost = max(1, len(line) // 4)
            if used + header_cost + line_cost > budget:
                continue
            selected_turns[turn_id] = candidate
            selected_sessions.add(session_key)
            session_rank[session_key] = min(session_rank.get(session_key, rank), rank)
            used += header_cost + line_cost

    grouped = defaultdict(list)
    for row in selected_turns.values():
        grouped[(row["session_index"], row["session_id"], row["session_time"])].append(row)
    parts: list[str] = []
    turn_ids: list[int] = []
    for (session_index, session_id, session_time), rows in sorted(
        grouped.items(), key=lambda item: (session_rank[item[0]], item[0][0])
    ):
        header = f"\n[Session {session_id} | {session_time or 'date unknown'}]\n"
        parts.append(header)
        for row in sorted(rows, key=lambda value: value["turn_index"]):
            line = f"{row['role']}: {row['content']}\n"
            parts.append(line)
            turn_ids.append(int(row["id"]))
    return "".join(parts), turn_ids
