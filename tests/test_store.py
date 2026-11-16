from src import store
from src.dialogue_gen import Turn, UserTimeline


def _manual_timeline(user_id: str) -> UserTimeline:
    """Hand-built, not generator-sampled, so the consolidation and deletion
    behavior under test is pinned to exact, known turns rather than whatever
    a random seed happens to produce."""
    turns = [
        Turn(0, 1, "home_city", "Seattle", False, "I live in Seattle these days."),
        Turn(0, 2, "job_title", "data analyst", False, "I work as a data analyst."),
        Turn(1, 3, "home_city", "Austin", False, "I live in Austin these days."),  # supersedes Seattle
        # a narrative gets built after session 1 lands at session%5==0 only at session 5,
        # so force an early one by calling build_narrative directly in the test below
        Turn(2, 4, "job_title", None, True, "Please forget what I told you about my job."),
    ]
    return UserTimeline(user_id=user_id, turns=turns)


def test_consolidation_keeps_latest_value_with_provenance(conn, embedder):
    tl = _manual_timeline("u_consolidate")
    with conn.cursor() as cur:
        store.ingest_timeline(cur, embedder, tl, apply_deletion_fix=True)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT value, provenance_ids FROM consolidated WHERE user_id=%s AND slot='home_city'", (tl.user_id,))
        value, provenance = cur.fetchone()
    assert value == "Austin"
    assert len(provenance) == 2  # both the Seattle and Austin raw facts fed consolidation


def test_deletion_removes_consolidated_row_and_tombstones_raw_facts(conn, embedder):
    tl = _manual_timeline("u_delete")
    with conn.cursor() as cur:
        store.ingest_timeline(cur, embedder, tl, apply_deletion_fix=True)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM consolidated WHERE user_id=%s AND slot='job_title'", (tl.user_id,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT COUNT(*) FROM raw_facts WHERE user_id=%s AND slot='job_title' AND deleted=TRUE", (tl.user_id,))
        assert cur.fetchone()[0] == 1  # the original statement, tombstoned


def test_deletion_leak_without_fix_then_fixed_by_propagation(conn, embedder):
    """Builds a narrative that captures job_title *before* the deletion
    request, then deletes job_title. Without the fix, the already-built
    narrative still names the deleted value (the leak). With the fix, the
    same narrative is redacted in place."""
    tl = _manual_timeline("u_leak")
    statement_turns = [t for t in tl.turns if not t.is_deletion_request]
    deletion_turn = next(t for t in tl.turns if t.is_deletion_request)

    with conn.cursor() as cur:
        for t in statement_turns:
            raw_id = store.ingest_raw(cur, embedder, tl.user_id, t)
            store.consolidate(cur, embedder, tl.user_id, t.slot, t.value, raw_id, t.turn_id, t.session_id)
        store.build_narrative(cur, embedder, tl.user_id, as_of_session=1)
        cur.execute("SELECT summary_text FROM narrative_summaries WHERE user_id=%s", (tl.user_id,))
        (before_text,) = cur.fetchone()
        assert "data analyst" in before_text

        store.delete_slot(cur, tl.user_id, deletion_turn.slot)
        cur.execute("SELECT summary_text, slots_included FROM narrative_summaries WHERE user_id=%s", (tl.user_id,))
        unfixed_text, unfixed_slots = cur.fetchone()
        assert "data analyst" in unfixed_text  # leak: redaction never ran
        assert "job_title" in unfixed_slots

        redacted_count = store.propagate_deletion(cur, embedder, tl.user_id, deletion_turn.slot)
        assert redacted_count == 1
        cur.execute("SELECT summary_text, slots_included, redacted FROM narrative_summaries WHERE user_id=%s", (tl.user_id,))
        fixed_text, fixed_slots, redacted = cur.fetchone()
        assert "data analyst" not in fixed_text
        assert "job_title" not in fixed_slots
        assert redacted is True
    conn.commit()


def test_decay_drops_slots_unmentioned_past_the_threshold(conn, embedder):
    from src import config

    tl = _manual_timeline("u_decay")
    statement_turns = [t for t in tl.turns if not t.is_deletion_request]
    with conn.cursor() as cur:
        for t in statement_turns:
            raw_id = store.ingest_raw(cur, embedder, tl.user_id, t)
            store.consolidate(cur, embedder, tl.user_id, t.slot, t.value, raw_id, t.turn_id, t.session_id)
        # home_city was last mentioned at session 1; well past the threshold
        # by session 1 + FORGET_AFTER_SESSIONS_UNMENTIONED + 1
        far_future_session = 1 + config.FORGET_AFTER_SESSIONS_UNMENTIONED + 1
        dropped = store.apply_decay(cur, tl.user_id, far_future_session)
        assert dropped == 2  # both home_city and job_title went quiet
        cur.execute("SELECT COUNT(*) FROM consolidated WHERE user_id=%s", (tl.user_id,))
        assert cur.fetchone()[0] == 0
    conn.commit()
