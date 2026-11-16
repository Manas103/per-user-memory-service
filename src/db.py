"""Postgres + pgvector connection and schema. One database, `public` schema,
four tables: raw_facts (append-only, every extracted fact including
deletion-request markers), consolidated (one current row per user+slot,
provenance = the raw_fact ids merged into it), narrative_summaries (periodic
per-user rollup paragraphs, the thing that actually drives the token-count
reduction), and never_forget_facts (the baseline: every raw fact, forever,
with no consolidation or narrative compaction, used only for the accuracy and
token-count comparison, not by the production-shaped store itself).
"""

from __future__ import annotations

import psycopg2

from src import config

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS raw_facts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id INT NOT NULL,
    turn_id INT NOT NULL,
    slot TEXT NOT NULL,
    value TEXT,                 -- NULL for a deletion-request row
    text TEXT NOT NULL,
    embedding vector({config.EMBED_DIM}) NOT NULL,
    is_deletion_request BOOLEAN NOT NULL DEFAULT FALSE,
    superseded BOOLEAN NOT NULL DEFAULT FALSE,
    deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS raw_facts_user_idx ON raw_facts (user_id);

CREATE TABLE IF NOT EXISTS consolidated (
    user_id TEXT NOT NULL,
    slot TEXT NOT NULL,
    value TEXT NOT NULL,
    provenance_ids INT[] NOT NULL,
    embedding vector({config.EMBED_DIM}) NOT NULL,
    last_turn_id INT NOT NULL,
    last_session_id INT NOT NULL,
    PRIMARY KEY (user_id, slot)
);

CREATE TABLE IF NOT EXISTS narrative_summaries (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    as_of_session INT NOT NULL,
    summary_text TEXT NOT NULL,
    slots_included TEXT[] NOT NULL,
    source_fact_ids INT[] NOT NULL,
    embedding vector({config.EMBED_DIM}) NOT NULL,
    redacted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS narrative_user_idx ON narrative_summaries (user_id);

CREATE TABLE IF NOT EXISTS never_forget_facts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    turn_id INT NOT NULL,
    slot TEXT NOT NULL,
    value TEXT,
    text TEXT NOT NULL,
    embedding vector({config.EMBED_DIM}) NOT NULL
);
CREATE INDEX IF NOT EXISTS nff_user_idx ON never_forget_facts (user_id);
"""


def get_conn():
    return psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        dbname=config.PG_DB,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
    )


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def reset_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS raw_facts, consolidated, narrative_summaries, never_forget_facts CASCADE")
    conn.commit()
    init_schema(conn)


def vec_literal(values) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"
