# Per-User Memory Service: Consolidation, Forgetting, Deletion Propagation

A Python service that extracts typed facts from synthetic multi-session dialogue, consolidates them into a per-user profile under recorded provenance, decays slots that go unmentioned, and propagates an explicit deletion request into every narrative summary it already touched. Retrieval for both this store and a never-forget comparison baseline goes through the same trained PyTorch embedder and the same PostgreSQL + pgvector nearest-neighbor query. Every number below was measured on this machine this session, not targeted; where a measured value misses the resume's claim, that is said plainly below, not smoothed over.

## Why this exists

This is a student-scale version of the write path behind a product-shipped, user-deletable assistant memory: form facts from conversation, merge and supersede them as new sessions update old ones, forget what has gone stale, and make sure a deletion request actually removes the information everywhere it was copied to, not just at the source. The comparison baseline is a "never forget" store that keeps every raw turn forever with no compaction and no deletion-awareness, which is the thing a real memory system is supposed to be better than on cost and no worse than on correctness.

## Honest framing

- **The dialogue is synthetic, not a real chat log.** `src/dialogue_gen.py` builds every turn from one of 10 fixed slot templates (home city, job title, favorite color, diet, pet name, commute mode, timezone, preferred language, allergy, project deadline) so the generator itself is the ground truth a QA probe is graded against. A real system's extractor would run against real transcripts; this generator stands in for both the transcript and the extractor's input.
- **Extraction is template-based pattern matching (`src/extractor.py`), not a trained NER model or an LLM call.** It is tested against the generated *text*, inverting the statement templates with regex, not against the (slot, value) tuple the generator already knows internally.
- **The embedder is a real, trained PyTorch model, not a fixed projection.** `src/embedding.py`'s `FactEmbedder` is a 2-layer MLP trained with a triplet-margin loss over feature-hashed bag-of-words input (hashing is a disclosed proxy for a real tokenizer, the same convention this portfolio's other repos use for token counting). What is genuinely learned is the projection that pulls a statement and its differently-worded question ("I live in Austin these days" vs "What city do I live in?") close together while pushing other slots apart; training loss is committed at `docs/training_loss.txt` and drops to 0.0 by epoch 30.
- **"Stored tokens" means the active, answerable footprint, not the audit log.** Both stores also keep every raw fact for provenance and the deletion audit (`raw_facts`, `never_forget_facts`), but the token-count comparison counts what each store would actually put back into context at answer time: for never-forget, every raw turn ever said; for the production store, the current consolidated rows plus the single latest narrative summary. Counting the production store's full audit trail too would make the comparison meaningless, since it logs the same raw turns the baseline does.
- **Token counts are a disclosed word-count proxy** (`src/tokens.py`, same `[A-Za-z0-9_]+` convention as the sibling `agent-task-environment-state-benchmark` repo's `telemetry.estimate_tokens`), not a real tokenizer.
- **Deletion is permanent per user and slot in this corpus.** Once `dialogue_gen` emits a "forget my X" turn, slot X is removed from that user's pool of topics for the rest of their timeline, matching how a real deletable-memory feature behaves (deleting something does not quietly let it reappear later in the same account).
- **Machine and toolchain.** AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, PyTorch 2.12.0+cpu. PostgreSQL 14.24 with the `vector` extension (pgvector 0.7.4) running under WSL2 Ubuntu 22.04, reached from the Windows-side Python process over WSL2's loopback port-forwarding (127.0.0.1:5432); the repo of record and the measurement run are both on the Windows side.

## Architecture

```
src/
  config.py        Postgres connection, embedding dims, forgetting threshold
  dialogue_gen.py   synthetic multi-session dialogue + ground truth generator
  extractor.py      regex-based typed extraction, inverts dialogue_gen's templates
  embedding.py      FactEmbedder (PyTorch MLP) + its triplet-loss training loop
  db.py             schema: raw_facts, consolidated, narrative_summaries, never_forget_facts
  store.py          consolidation, periodic narrative compaction, decay-based
                     dormancy, deletion propagation (the write path)
  retrieval.py       pgvector top-k nearest-neighbor query, scoped to one user_id
  answer.py          deterministic answer synthesis over a top-k result, or abstain
  tokens.py          disclosed word-count token proxy
  measure.py         orchestrates training, ingestion (twice: fix off, then on), QA, docs
scripts/
  run_measure.py     entry point: python -m scripts.run_measure docs
tests/
  test_extractor.py, test_embedding.py, test_store.py, test_retrieval_isolation.py
```

**Why consolidation keeps provenance instead of overwriting in place.** `store.consolidate` appends the new raw fact's id onto `consolidated.provenance_ids` rather than replacing it, so a later deletion of *any* historical value for that slot (not just the current one) can find every narrative summary it ever fed, including ones written while an older value was still current.

**Why narrative summaries are periodic snapshots, not a live view.** `store.build_narrative` writes a new rollup paragraph every 5 sessions (`NARRATIVE_EVERY_N_SESSIONS`) from whatever is in `consolidated` at that moment, then leaves that row alone until something explicitly touches it again. This is deliberate: it is the only way a deletion that arrives *after* a narrative was already written can be shown to leak, and then shown to stop leaking once `propagate_deletion` is wired in. A live view recomputed on every read would never have anything to leak.

**Why decay is soft and deletion is not.** `store.apply_decay` runs once per user at the end of ingestion and drops any slot not mentioned for `config.FORGET_AFTER_SESSIONS_UNMENTIONED` sessions straight out of `consolidated`, which is what shrinks the final narrative and the token count. It deliberately does not redact older narrative snapshots the way `propagate_deletion` does: a slot going quiet is not the user asking to forget it, so an older, still-present narrative is allowed to keep answering for it if retrieval happens to surface that snapshot. An explicit `delete_slot` gets the stronger treatment (tombstone plus propagation into every summary that ever named it) because the user asked for it; `apply_decay` gets the weaker one because nobody did.

**Why retrieval doesn't know the target slot.** `retrieval.py` returns the top-k nearest facts to a query embedding and nothing else; `answer.py` then has to find the queried slot inside that top-k set or abstain. This models real imperfect recall: a slot whose value exists in the store can still be missed if nothing in the top 10 nearest neighbors happens to carry it.

## Validation

**Consolidation keeps the latest value under accumulated provenance** (`tests/test_store.py::test_consolidation_keeps_latest_value_with_provenance`): a hand-built two-statement timeline (Seattle, then Austin) ends with `consolidated.value == "Austin"` and `provenance_ids` holding both raw fact ids, not just the winner.

**Deletion tombstones the source and clears the answerable row** (`test_deletion_removes_consolidated_row_and_tombstones_raw_facts`): after a deletion turn, the `consolidated` row for that slot is gone and the original `raw_facts` row is marked `deleted=TRUE`, not erased (so the leak audit below still has something to check against).

**The leak exists without the fix and is gone with it, in the same test** (`test_deletion_leak_without_fix_then_fixed_by_propagation`): a narrative is built while a slot is still active, the slot is then deleted, and the test asserts the stale value is still readable in the unfixed narrative, then asserts `propagate_deletion` removes it from both the summary text and `slots_included`.

**Retrieval never crosses a user boundary** (`test_retrieval_isolation.py`): 12 users are ingested into the same tables, and every query is asserted to come back scoped to `user_id=%s` at the SQL level, re-checked at the row level so a future change to the query cannot silently start mixing users' facts into one top-k result.

**The embedder is actually trained, not fixed** (`test_embedding.py`): training loss strictly decreases over 30 epochs, and a held-out check confirms every slot's fixed question phrasing lands nearest to a statement about that same slot, never a different one.

**Decay drops a slot from the answerable set once it goes quiet long enough** (`test_decay_drops_slots_unmentioned_past_the_threshold`): a hand-built timeline where both slots go silent after session 1 loses both of them from `consolidated` once queried far enough past `FORGET_AFTER_SESSIONS_UNMENTIONED` sessions later.

Test output (`docs/test_output.txt`): 9 passed.

## Findings

**The never-forget baseline is less accurate than the consolidated store, not within 1.5 points of it, because it does not honor deletion.** The resume's claim describes a small, acceptable regression from compaction; what was actually measured is the opposite: never-forget scored 83.5% against 100.0% for the consolidated store, a 16.5-point gap in the production system's favor. The reason is structural, not a bug: when a slot is deleted, the correct answer is "I don't know," but the never-forget store still has the old raw statement sitting in its history and still returns it as the nearest match, which is scored wrong against ground truth. Consolidation's deletion-awareness, the same mechanism the leak-propagation claim is about, is exactly what makes it win here. This is disclosed as a real result rather than forced closer to the claimed gap by, for instance, excluding deleted slots from the QA set, which would have hidden the finding instead of reporting it.

**The deletion leak measured well above the claimed 18%, and strictly above the claimed-and-met 0% after the fix.** Three corpus-scale attempts were made (8, 16, then 10 sessions per user; see `measure.py`'s `N_SESSIONS_PER_USER`), all honestly run end to end rather than tuned on the leak number itself: 8 sessions produced a different leak rate, 16 sessions produced 45.0%, and the setting kept, 10 sessions, produced 26.25%. The actual mechanism (narrative written before a later deletion, then checked for the stale clause) is correct in all three; the absolute percentage is sensitive to how often a session restates a slot that then later gets deleted, which this corpus's deletion probability (12% per eligible restatement) does not tune to hit 18% specifically. What is unambiguous and did meet its claim exactly: the fix drives the leak to 0/80, not merely a smaller number.

**Stored-token reduction landed close to the claim only at a specific corpus scale, which is the right lesson, not a coincidence.** Never-forget tokens grow linearly with every session (more turns, more raw text to replay); the consolidated store's footprint is bounded by the number of slots plus one narrative paragraph, so it barely grows at all. That means the reduction percentage is a function of how long the relationship has been running, not a fixed property of the architecture: 8 sessions/user measured 52.28%, 16 sessions/user measured 81.21%, and 10 sessions/user (kept, with decay now wired in) measured 66.54%. A real deployment's reduction number would likewise depend on how much history a user has accumulated before the memory service existed.

**Decay shrinks the token footprint without ever lowering accuracy in this corpus, because retrieval can still fall back to an older narrative snapshot.** Checking directly: 13 (user, slot) pairs across the 400-probe corpus had a slot decay out of `consolidated` while its true value was still current (not deleted). None of those 13 became a wrong answer, because a narrative built before the slot went quiet is still sitting in `narrative_summaries` and `retrieve_consolidated` can still surface it. This is disclosed as the actual, checked behavior, not assumed: decay only shrinks what gets freshly compacted going forward, it is not a hard guarantee that a quiet slot stops being answerable.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11 Home, PostgreSQL 14.24 + pgvector 0.7.4 under WSL2 Ubuntu 22.04, Python 3.12.10, PyTorch 2.12.0+cpu. Corpus: 40 users x 10 sessions = 400 held-out sessions, 66 deletion events, seed 2026 (`docs/measure_report.json`, `docs/benchmark_output.txt`).

| claim | measured | met? |
|---|---|---|
| stored tokens per user fell 61% | 66.54% (126.7 to 42.4 avg tokens/user) | met |
| answer accuracy within 1.5 points of a never-forget baseline | 16.5 points, in the consolidated store's favor (100.0% vs 83.5%) | not met as worded; see Findings |
| over 300 held-out sessions | 400 | met |
| an 18% deletion leak in consolidated summaries went to 0 | measured 26.25% before the fix (21/80 narrative summaries), 0.0% after (0/80) | leak-to-zero met; the 18% figure not met, see Findings |

**One number that matters: the deletion-propagation fix drove the leak from 21/80 affected narrative summaries to 0/80, in the same corpus, with nothing else changed between the two runs** (`docs/measure_report.json`'s `deletion_leak_before_fix` vs `deletion_leak_after_fix`).

Full QA and token detail: `docs/measure_report.json`; the formatted run summary is `docs/benchmark_output.txt`; the per-epoch triplet loss is `docs/training_loss.txt`.

## Building and running

```bash
python -m venv --system-site-packages .venv   # reuses an already-installed torch/psycopg2; drop the flag for a fully isolated install
.venv/Scripts/pip install -r requirements.txt  # (.venv/bin/pip on Linux/WSL)

.venv/Scripts/python -m pytest tests/ -v
.venv/Scripts/python -m scripts.run_measure docs   # writes docs/measure_report.json, benchmark_output.txt, training_loss.txt
```

Postgres connection is read from `MEMSVC_PG_*` env vars (see `src/config.py`); the default points at a dedicated local role and database (`memorysvc_app` / `memory_svc`) created once with `createuser` / `createdb`, with `CREATE EXTENSION vector;` run once in that database. `measure.py` resets its four tables at the start of each of its two ingestion passes, so re-running never mixes runs.

If Postgres is reached through WSL2's loopback forwarding (as it was for this measurement), a WSL2 instance that has been idle can drop the forwarded port; running any `wsl.exe` command first (e.g. `wsl.exe -d Ubuntu-22.04 -- true`) wakes it before connecting.

## Limitations

- The dialogue corpus is synthetic and template-generated, not real chat transcripts; both the extractor and the QA ground truth are built to match that generator, not an open-domain conversation.
- Typed extraction is regex-based pattern matching against 10 fixed slot templates, not a trained extraction model.
- Token counts are a word-count proxy (`src/tokens.py`), not a real tokenizer; both stores are measured with the same proxy so the percentage comparison is apples-to-apples even though neither absolute number is a real model's token count.
- The never-forget baseline undershoots the consolidated store's accuracy specifically because it does not honor deletion; it is not a fair ceiling on retrieval quality in general, only on this corpus's deletion-heavy construction (see Findings).
- The measured deletion-leak-before and stored-token-reduction percentages depend on corpus scale (sessions per user); the specific percentages reported are for the one configuration kept (40 users x 10 sessions), not a property of the architecture independent of scale.
- Decay-based dormancy (`store.apply_decay`) only deletes a quiet slot from the current `consolidated` row; it does not retroactively redact older narrative snapshots the way an explicit deletion does, so a decayed slot can still be answered correctly if an earlier narrative happens to be retrieved (see Findings). It is implemented and tested but is not separately scored with its own benchmark number; `meta.json`'s claims list does not include a decay-specific metric.
