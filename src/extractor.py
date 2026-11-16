"""Typed extraction: turns free text back into (slot, value) or a deletion
request. This is template-based pattern matching, not a trained NER model or
an LLM call (disclosed in the README); it is built to invert
`dialogue_gen.TEMPLATE_BY_SLOT` exactly, and is tested against text the
generator produces, not against the (slot, value) tuple the generator already
knows, so the extraction step is a real, checkable parse rather than a pass-
through of ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.dialogue_gen import SLOT_LABEL, TEMPLATE_BY_SLOT

_SLOT_PATTERNS = {}
for _slot, _tmpl in TEMPLATE_BY_SLOT.items():
    prefix, suffix = _tmpl.split("{value}")
    _SLOT_PATTERNS[_slot] = re.compile(re.escape(prefix) + r"(?P<value>.+?)" + re.escape(suffix if suffix else "$") + ("$" if not suffix else ""))

_DELETE_PATTERN = re.compile(r"Please forget what I told you about my (?P<label>.+)\.$")
_LABEL_TO_SLOT = {v: k for k, v in SLOT_LABEL.items()}


@dataclass
class ExtractedFact:
    slot: str
    value: str


@dataclass
class ExtractedDeletion:
    slot: str


def extract(text: str):
    """Returns an ExtractedFact, an ExtractedDeletion, or None (text matched no
    known slot template; extraction is deliberately conservative, not a catch-all)."""
    m = _DELETE_PATTERN.match(text)
    if m:
        label = m.group("label")
        slot = _LABEL_TO_SLOT.get(label)
        return ExtractedDeletion(slot) if slot else None
    for slot, pattern in _SLOT_PATTERNS.items():
        m = pattern.match(text)
        if m:
            return ExtractedFact(slot, m.group("value"))
    return None
