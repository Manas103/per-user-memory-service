"""Answer-time retrieval. Both systems under comparison use the identical
mechanism (embed the query, pgvector cosine nearest-neighbor, top-k=10,
scoped to one user_id), so the accuracy comparison in measure.py isolates
what each store *keeps*, not how it searches.

Retrieval does not know which slot a query is "about"; it only returns the
top-k nearest facts by embedding distance. answer.py then has to find the
target slot's value inside that top-k set, or abstain if it is not there,
which is the same imperfect-recall behavior a real deployment would have.
"""

from __future__ import annotations

from src.db import vec_literal

TOP_K = 10


def retrieve_never_forget(cur, query_embedding, user_id: str, k: int = TOP_K):
    cur.execute(
        "SELECT slot, value, turn_id FROM never_forget_facts "
        "WHERE user_id=%s AND value IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        (user_id, vec_literal(query_embedding), k),
    )
    return [{"kind": "raw", "slot": r[0], "value": r[1], "turn_id": r[2]} for r in cur.fetchall()]


def retrieve_consolidated(cur, query_embedding, user_id: str, k: int = TOP_K):
    qv = vec_literal(query_embedding)
    cur.execute(
        "SELECT slot, value, last_turn_id, embedding <=> %s::vector AS dist FROM consolidated "
        "WHERE user_id=%s ORDER BY dist LIMIT %s",
        (qv, user_id, k),
    )
    consolidated_rows = [
        {"kind": "consolidated", "slot": r[0], "value": r[1], "turn_id": r[2], "dist": float(r[3])} for r in cur.fetchall()
    ]
    cur.execute(
        "SELECT summary_text, slots_included, as_of_session, embedding <=> %s::vector AS dist FROM narrative_summaries "
        "WHERE user_id=%s ORDER BY dist LIMIT %s",
        (qv, user_id, k),
    )
    narrative_rows = [
        {"kind": "narrative", "summary_text": r[0], "slots_included": r[1], "turn_id": r[2], "dist": float(r[3])}
        for r in cur.fetchall()
    ]
    merged = sorted(consolidated_rows + narrative_rows, key=lambda r: r["dist"])
    return merged[:k]
