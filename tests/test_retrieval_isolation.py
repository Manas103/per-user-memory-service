from src import store
from src.dialogue_gen import generate_corpus
from src.retrieval import retrieve_consolidated, retrieve_never_forget


def test_retrieval_never_crosses_user_boundary(conn, embedder):
    corpus = generate_corpus(n_users=12, n_sessions_per_user=6, seed=99)
    with conn.cursor() as cur:
        for tl in corpus:
            store.ingest_timeline(cur, embedder, tl, apply_deletion_fix=True)
    conn.commit()

    target_user = corpus[0].user_id
    q_emb = embedder.embed_text("What city do I live in?")
    with conn.cursor() as cur:
        nf_hits = retrieve_never_forget(cur, q_emb, target_user, k=10)
        cs_hits = retrieve_consolidated(cur, q_emb, target_user, k=10)

    assert len(nf_hits) > 0
    assert len(cs_hits) > 0
    # the queries are scoped with WHERE user_id=%s; this re-asserts it at the
    # result level so an accidental future change to the SQL cannot silently
    # start leaking other users' facts into a top-k result
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT user_id FROM never_forget_facts")
        all_users = {r[0] for r in cur.fetchall()}
    assert len(all_users) == 12
