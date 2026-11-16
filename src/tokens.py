"""Token-count proxy, same convention as the sibling
`agent-task-environment-state-benchmark` repo's `telemetry.estimate_tokens`:
word-like substrings in the text, not a real tokenizer. Disclosed in the
README; used identically for both stores being compared so the percentage
reduction is apples-to-apples even though neither absolute number is a real
model's token count.
"""

import re

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))
