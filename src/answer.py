"""Deterministic answer synthesis over a top-k retrieval result. No LLM is
called; the point of this benchmark is to score what the memory *store*
surfaces, not a generator's ability to phrase a sentence (disclosed, see
README). A target slot's value is read directly off a matching candidate's
structured field (consolidated, raw) or parsed out of a narrative summary's
text with the same regex family extractor.py uses.
"""

from __future__ import annotations

import re

from src.dialogue_gen import SLOT_LABEL

ABSTAIN = None


def _parse_narrative_clause(summary_text: str, slot: str) -> str | None:
    label = SLOT_LABEL[slot]
    m = re.search(re.escape(label) + r" is (?P<value>[^;.]+)", summary_text)
    return m.group("value") if m else None


def synthesize_never_forget(candidates, target_slot: str) -> str | None:
    matches = [c for c in candidates if c["kind"] == "raw" and c["slot"] == target_slot]
    if not matches:
        return ABSTAIN
    return max(matches, key=lambda c: c["turn_id"])["value"]


def synthesize_consolidated(candidates, target_slot: str) -> str | None:
    for c in candidates:
        if c["kind"] == "consolidated" and c["slot"] == target_slot:
            return c["value"]
    for c in candidates:
        if c["kind"] == "narrative" and target_slot in c["slots_included"]:
            value = _parse_narrative_clause(c["summary_text"], target_slot)
            if value is not None:
                return value
    return ABSTAIN
