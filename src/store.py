"""The write path: typed extraction already happened (src/extractor.py); this
module owns consolidation (merge and supersede under recorded provenance),
periodic narrative compaction (the thing that actually shrinks token count),
decay-based dormancy, and deletion propagation.

Two independent record-keeping tracks share one database:
  - raw_facts / consolidated / narrative_summaries: the system under test.
  - never_forget_facts: every raw turn, forever, no consolidation, no
    compaction, no deletion-awareness. Used only as the comparison baseline
    in measure.py; nothing in the system under test reads from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.db import vec_literal
from src.dialogue_gen import SLOT_LABEL, Turn, UserTimeline

NARRATIVE_EVERY_N_SESSIONS = 5


def ingest_never_forget(cur, embedder, user_id: str, turn: Turn) -> None:
    emb = embedder.embed_text(turn.text)
    cur.execute(
        "INSERT INTO never_forget_facts (user_id, turn_id, slot, value, text, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s::vector)",
        (user_id, turn.turn_id, turn.slot, turn.value, turn.text, vec_literal(emb)),
    )


def ingest_raw(cur, embedder, user_id: str, turn: Turn) -> int:
    emb = embedder.embed_text(turn.text)
    cur.execute(
        "INSERT INTO raw_facts (user_id, session_id, turn_id, slot, value, text, embedding, is_deletion_request) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s) RETURNING id",
        (user_id, turn.session_id, turn.turn_id, turn.slot, turn.value, turn.text, vec_literal(emb), turn.is_deletion_request),
    )
    return cur.fetchone()[0]


def consolidate(cur, embedder, user_id: str, slot: str, value: str, raw_fact_id: int, turn_id: int, session_id: int) -> None:
    text = f"{SLOT_LABEL[slot]}: {value}"
    emb = embedder.embed_text(text)
    cur.execute(
        """
        INSERT INTO consolidated (user_id, slot, value, provenance_ids, embedding, last_turn_id, last_session_id)
        VALUES (%s, %s, %s, ARRAY[%s], %s::vector, %s, %s)
        ON CONFLICT (user_id, slot) DO UPDATE SET
            value = EXCLUDED.value,
            provenance_ids = consolidated.provenance_ids || EXCLUDED.provenance_ids,
            embedding = EXCLUDED.embedding,
            last_turn_id = EXCLUDED.last_turn_id,
            last_session_id = EXCLUDED.last_session_id
        """,
        (user_id, slot, value, raw_fact_id, vec_literal(emb), turn_id, session_id),
    )


def apply_decay(cur, user_id: str, as_of_session: int) -> int:
    """Decay-based forgetting: a slot not mentioned again for
    `config.FORGET_AFTER_SESSIONS_UNMENTIONED` sessions drops out of the
    answerable consolidated set. This is a quieter mechanism than
    `delete_slot`: no tombstone, no deletion-propagation audit, because the
    user never asked for it to be forgotten, it simply went stale. Returns
    the number of slots dropped."""
    cur.execute(
        "DELETE FROM consolidated WHERE user_id=%s AND (%s - last_session_id) > %s",
        (user_id, as_of_session, config.FORGET_AFTER_SESSIONS_UNMENTIONED),
    )
    return cur.rowcount


def delete_slot(cur, user_id: str, slot: str) -> None:
    """Explicit user deletion: remove the slot from the answerable consolidated
    set and tombstone every raw fact ever recorded for that user+slot. Does
    NOT touch narrative_summaries; that is propagate_deletion's job, run (or
    not run) separately so measure.py can show the leak before the fix."""
    cur.execute("DELETE FROM consolidated WHERE user_id=%s AND slot=%s", (user_id, slot))
    cur.execute("UPDATE raw_facts SET deleted=TRUE WHERE user_id=%s AND slot=%s", (user_id, slot))


def build_narrative(cur, embedder, user_id: str, as_of_session: int) -> None:
    cur.execute(
        "SELECT slot, value, provenance_ids FROM consolidated WHERE user_id=%s ORDER BY slot",
        (user_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return
    clauses = [f"{SLOT_LABEL[slot]} is {value}" for slot, value, _ in rows]
    summary_text = f"Profile for {user_id} as of session {as_of_session}: " + "; ".join(clauses) + "."
    slots_included = [r[0] for r in rows]
    source_fact_ids = [pid for r in rows for pid in r[2]]
    emb = embedder.embed_text(summary_text)
    cur.execute(
        "INSERT INTO narrative_summaries (user_id, as_of_session, summary_text, slots_included, source_fact_ids, embedding) "
        "VALUES (%s, %s, %s, %s, %s, %s::vector)",
        (user_id, as_of_session, summary_text, slots_included, source_fact_ids, vec_literal(emb)),
    )


def propagate_deletion(cur, embedder, user_id: str, slot: str) -> int:
    """The fix: find every narrative summary that baked in a now-deleted
    slot's clause and redact it (regenerate the text without that clause),
    rather than leaving the stale value sitting in an already-written
    summary. Returns the number of summaries redacted."""
    cur.execute(
        "SELECT id, summary_text, slots_included FROM narrative_summaries "
        "WHERE user_id=%s AND %s = ANY(slots_included)",
        (user_id, slot),
    )
    rows = cur.fetchall()
    for summary_id, _text, slots_included in rows:
        remaining = [s for s in slots_included if s != slot]
        cur.execute("SELECT slot, value FROM consolidated WHERE user_id=%s AND slot = ANY(%s) ORDER BY slot", (user_id, remaining))
        remaining_rows = cur.fetchall()
        clauses = [f"{SLOT_LABEL[s]} is {v}" for s, v in remaining_rows]
        new_text = f"Profile for {user_id} (redacted): " + ("; ".join(clauses) + "." if clauses else "no active slots.")
        emb = embedder.embed_text(new_text)
        cur.execute(
            "UPDATE narrative_summaries SET summary_text=%s, slots_included=%s, embedding=%s::vector, redacted=TRUE WHERE id=%s",
            (new_text, remaining, vec_literal(emb), summary_id),
        )
    return len(rows)


def ingest_timeline(cur, embedder, tl: UserTimeline, apply_deletion_fix: bool) -> None:
    sessions_seen = -1
    for turn in tl.turns:
        ingest_never_forget(cur, embedder, tl.user_id, turn)
        if turn.is_deletion_request:
            delete_slot(cur, tl.user_id, turn.slot)
            ingest_raw(cur, embedder, tl.user_id, turn)
            if apply_deletion_fix:
                propagate_deletion(cur, embedder, tl.user_id, turn.slot)
        else:
            raw_id = ingest_raw(cur, embedder, tl.user_id, turn)
            consolidate(cur, embedder, tl.user_id, turn.slot, turn.value, raw_id, turn.turn_id, turn.session_id)
        if turn.session_id != sessions_seen:
            sessions_seen = turn.session_id
            if sessions_seen > 0 and sessions_seen % NARRATIVE_EVERY_N_SESSIONS == 0:
                build_narrative(cur, embedder, tl.user_id, sessions_seen)
    # final narrative so every user has at least one, even a short timeline
    final_session = tl.turns[-1].session_id if tl.turns else 0
    apply_decay(cur, tl.user_id, final_session)
    build_narrative(cur, embedder, tl.user_id, final_session)
