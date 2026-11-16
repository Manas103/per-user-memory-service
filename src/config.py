"""Static configuration. Nothing here is fit to a target number; thresholds were
picked once from the dialogue generator's own scale (number of slots, sessions)
and then left alone while measure.py ran.
"""

import os

PG_HOST = os.environ.get("MEMSVC_PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("MEMSVC_PG_PORT", "5432"))
PG_DB = os.environ.get("MEMSVC_PG_DB", "memory_svc")
PG_USER = os.environ.get("MEMSVC_PG_USER", "memorysvc_app")
PG_PASSWORD = os.environ.get("MEMSVC_PG_PASSWORD", "memorysvc_pw")

EMBED_DIM = 32
HASH_DIM = 256  # bag-of-hashed-words input width before the learned projection

# A slot is forgotten (dropped from the active, answerable set) once it has gone
# this many sessions without being mentioned again and the user has not pinned it.
FORGET_AFTER_SESSIONS_UNMENTIONED = 6

# Consolidation keeps, per (user, slot), the current value plus the ids of every
# raw fact that was merged into it (provenance), so a later deletion of any one
# of those raw facts can find every summary it touched.
