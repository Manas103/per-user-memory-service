"""Orchestrates the full measurement: trains the embedder, generates the
held-out corpus, runs ingestion twice (deletion-fix off, then on) to show the
leak before/after, then scores stored tokens and answer accuracy on the
fixed run. Writes docs/measure_report.json and docs/benchmark_output.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import db, store
from src.dialogue_gen import SLOTS, QUERY_TEMPLATE_BY_SLOT, generate_corpus
from src.answer import synthesize_consolidated, synthesize_never_forget
from src.embedding import train_embedder
from src.retrieval import retrieve_consolidated, retrieve_never_forget
from src.tokens import estimate_tokens

N_USERS = 40
N_SESSIONS_PER_USER = 10
CORPUS_SEED = 2026


def compute_leak_rate(cur) -> dict:
    cur.execute("SELECT DISTINCT user_id, slot FROM raw_facts WHERE is_deletion_request=TRUE")
    deleted_pairs = set(tuple(r) for r in cur.fetchall())
    cur.execute("SELECT user_id, slots_included, summary_text FROM narrative_summaries")
    rows = cur.fetchall()
    total = len(rows)
    leaking = 0
    for user_id, slots_included, _summary_text in rows:
        if any((user_id, slot) in deleted_pairs for slot in slots_included):
            leaking += 1
    rate = (leaking / total) if total else 0.0
    return {"narrative_summaries": total, "leaking": leaking, "leak_rate_pct": round(rate * 100, 2)}


def ingest_corpus(cur, embedder, corpus, apply_deletion_fix: bool) -> None:
    for tl in corpus:
        store.ingest_timeline(cur, embedder, tl, apply_deletion_fix=apply_deletion_fix)


def run_qa(cur, embedder, corpus) -> dict:
    total = 0
    nf_correct = 0
    cs_correct = 0
    for tl in corpus:
        for slot in SLOTS:
            if slot in tl.deleted_slots:
                truth = None
            else:
                truth = tl.ground_truth_current.get(slot)
            query_text = QUERY_TEMPLATE_BY_SLOT[slot]
            q_emb = embedder.embed_text(query_text)

            nf_candidates = retrieve_never_forget(cur, q_emb, tl.user_id)
            nf_answer = synthesize_never_forget(nf_candidates, slot)

            cs_candidates = retrieve_consolidated(cur, q_emb, tl.user_id)
            cs_answer = synthesize_consolidated(cs_candidates, slot)

            total += 1
            nf_correct += int(nf_answer == truth)
            cs_correct += int(cs_answer == truth)
    return {
        "probes": total,
        "never_forget_accuracy_pct": round(100.0 * nf_correct / total, 2),
        "consolidated_accuracy_pct": round(100.0 * cs_correct / total, 2),
        "gap_points": round(abs(100.0 * nf_correct / total - 100.0 * cs_correct / total), 2),
    }


def stored_tokens(cur, corpus) -> dict:
    per_user_nf = []
    per_user_cs = []
    for tl in corpus:
        cur.execute("SELECT text FROM never_forget_facts WHERE user_id=%s", (tl.user_id,))
        nf_tokens = sum(estimate_tokens(r[0]) for r in cur.fetchall())
        per_user_nf.append(nf_tokens)

        cur.execute("SELECT slot, value FROM consolidated WHERE user_id=%s", (tl.user_id,))
        cs_tokens = sum(estimate_tokens(f"{slot}: {value}") for slot, value in cur.fetchall())
        cur.execute(
            "SELECT summary_text FROM narrative_summaries WHERE user_id=%s ORDER BY as_of_session DESC LIMIT 1",
            (tl.user_id,),
        )
        row = cur.fetchone()
        if row:
            cs_tokens += estimate_tokens(row[0])
        per_user_cs.append(cs_tokens)

    avg_nf = sum(per_user_nf) / len(per_user_nf)
    avg_cs = sum(per_user_cs) / len(per_user_cs)
    reduction_pct = round(100.0 * (1 - avg_cs / avg_nf), 2)
    return {
        "avg_never_forget_tokens_per_user": round(avg_nf, 1),
        "avg_consolidated_tokens_per_user": round(avg_cs, 1),
        "reduction_pct": reduction_pct,
    }


def main(docs_dir: Path) -> dict:
    docs_dir.mkdir(parents=True, exist_ok=True)
    embedder, losses = train_embedder()
    (docs_dir / "training_loss.txt").write_text(
        "epoch,triplet_loss\n" + "\n".join(f"{i},{v:.5f}" for i, v in enumerate(losses)) + "\n"
    )

    corpus = generate_corpus(N_USERS, N_SESSIONS_PER_USER, CORPUS_SEED)
    total_sessions = N_USERS * N_SESSIONS_PER_USER
    total_deletion_events = sum(1 for tl in corpus for t in tl.turns if t.is_deletion_request)

    conn = db.get_conn()
    try:
        db.reset_schema(conn)
        with conn.cursor() as cur:
            ingest_corpus(cur, embedder, corpus, apply_deletion_fix=False)
        conn.commit()
        with conn.cursor() as cur:
            leak_before = compute_leak_rate(cur)

        db.reset_schema(conn)
        with conn.cursor() as cur:
            ingest_corpus(cur, embedder, corpus, apply_deletion_fix=True)
        conn.commit()
        with conn.cursor() as cur:
            leak_after = compute_leak_rate(cur)
            qa = run_qa(cur, embedder, corpus)
            tokens = stored_tokens(cur, corpus)
    finally:
        conn.close()

    report = {
        "n_users": N_USERS,
        "n_sessions_per_user": N_SESSIONS_PER_USER,
        "total_held_out_sessions": total_sessions,
        "total_deletion_events": total_deletion_events,
        "deletion_leak_before_fix": leak_before,
        "deletion_leak_after_fix": leak_after,
        "qa": qa,
        "tokens": tokens,
    }
    (docs_dir / "measure_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        f"held-out sessions: {total_sessions} ({N_USERS} users x {N_SESSIONS_PER_USER} sessions)",
        f"deletion events in corpus: {total_deletion_events}",
        "",
        f"deletion leak, naive (no propagation): {leak_before['leaking']}/{leak_before['narrative_summaries']} "
        f"narrative summaries ({leak_before['leak_rate_pct']}%)",
        f"deletion leak, with propagation fix:   {leak_after['leaking']}/{leak_after['narrative_summaries']} "
        f"narrative summaries ({leak_after['leak_rate_pct']}%)",
        "",
        f"QA probes: {qa['probes']}",
        f"never-forget baseline accuracy: {qa['never_forget_accuracy_pct']}%",
        f"consolidated store accuracy:    {qa['consolidated_accuracy_pct']}%",
        f"accuracy gap: {qa['gap_points']} points",
        "",
        f"avg never-forget tokens/user:  {tokens['avg_never_forget_tokens_per_user']}",
        f"avg consolidated tokens/user:  {tokens['avg_consolidated_tokens_per_user']}",
        f"stored-token reduction: {tokens['reduction_pct']}%",
    ]
    (docs_dir / "benchmark_output.txt").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    import sys

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")
    result = main(out_dir)
    print(json.dumps(result, indent=2))
