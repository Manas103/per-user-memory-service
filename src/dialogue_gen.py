"""Synthetic multi-session dialogue generator.

There is no real chat log here (disclosed, see README): each "turn" is a
sentence built from one of 10 typed slot templates, so that the generator
itself knows the ground-truth (user, slot, value, session) a later QA probe
can be graded against without an LLM judge. A real deployment would run typed
extraction over a real transcript; this generator stands in for both the
transcript and the extractor's input, which is why it, not an LLM, owns the
ground truth.

Within a user's session sequence:
  - most sessions restate or introduce a slot's value unchanged
  - some sessions *update* a slot to a new value (a contradiction the store
    must resolve to the newer value, not the union of both)
  - some sessions issue an explicit deletion request for a slot already stated
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SLOTS = [
    "home_city",
    "job_title",
    "favorite_color",
    "dietary_restriction",
    "pet_name",
    "commute_mode",
    "timezone",
    "preferred_language",
    "allergy",
    "project_deadline",
]

VALUE_POOLS = {
    "home_city": ["Seattle", "Austin", "Denver", "Boston", "Raleigh", "Portland", "Minneapolis", "Tampa"],
    "job_title": ["data analyst", "product manager", "backend engineer", "research assistant", "designer"],
    "favorite_color": ["teal", "crimson", "navy", "amber", "forest green", "slate gray"],
    "dietary_restriction": ["vegetarian", "gluten-free", "pescatarian", "none", "lactose-free"],
    "pet_name": ["Mochi", "Biscuit", "Juniper", "Pixel", "Rocco", "Sable"],
    "commute_mode": ["bike", "train", "car", "walk", "bus"],
    "timezone": ["US/Pacific", "US/Eastern", "US/Central", "US/Mountain"],
    "preferred_language": ["English", "Spanish", "Mandarin", "Hindi", "Portuguese"],
    "allergy": ["peanuts", "shellfish", "penicillin", "none", "pollen"],
    "project_deadline": ["March 3", "April 14", "May 9", "June 22", "July 30"],
}

TEMPLATE_BY_SLOT = {
    "home_city": "I live in {value} these days.",
    "job_title": "I work as a {value}.",
    "favorite_color": "My favorite color is {value}.",
    "dietary_restriction": "I'm {value}, just so you know.",
    "pet_name": "My pet's name is {value}.",
    "commute_mode": "I usually commute by {value}.",
    "timezone": "I'm in the {value} timezone.",
    "preferred_language": "I'd rather we talk in {value}.",
    "allergy": "I'm allergic to {value}.",
    "project_deadline": "My project deadline is {value}.",
}

DELETE_TEMPLATE = "Please forget what I told you about my {slot_label}."

# Fixed question phrasing used at answer time; deliberately worded differently
# from TEMPLATE_BY_SLOT's statements (fewer shared surface words) so retrieval
# is actually tested on semantic closeness, not string overlap. These also
# feed embedding.py's training as same-slot positives for the statement they
# pair with.
QUERY_TEMPLATE_BY_SLOT = {
    "home_city": "What city do I live in?",
    "job_title": "What do I do for work?",
    "favorite_color": "What's my favorite color?",
    "dietary_restriction": "What's my diet like?",
    "pet_name": "What's my pet's name?",
    "commute_mode": "How do I get to work?",
    "timezone": "What timezone am I in?",
    "preferred_language": "Which language do I prefer?",
    "allergy": "What am I allergic to?",
    "project_deadline": "When is my project deadline?",
}

SLOT_LABEL = {
    "home_city": "home city",
    "job_title": "job",
    "favorite_color": "favorite color",
    "dietary_restriction": "diet",
    "pet_name": "pet's name",
    "commute_mode": "commute",
    "timezone": "timezone",
    "preferred_language": "preferred language",
    "allergy": "allergy",
    "project_deadline": "project deadline",
}


@dataclass
class Turn:
    session_id: int
    turn_id: int
    slot: str
    value: str | None  # None for a deletion-request turn
    is_deletion_request: bool = False
    text: str = ""


@dataclass
class UserTimeline:
    user_id: str
    turns: list[Turn] = field(default_factory=list)
    ground_truth_current: dict = field(default_factory=dict)  # slot -> value as of the last turn, or None if deleted
    deleted_slots: set = field(default_factory=set)


def generate_user_timeline(user_id: str, n_sessions: int, rng: random.Random) -> UserTimeline:
    tl = UserTimeline(user_id=user_id)
    turn_counter = 0
    # each user has a fixed subset of slots they ever talk about, so some QA
    # probes are legitimately answerable with "I don't know" (abstention)
    active_slots = list(rng.sample(SLOTS, k=rng.randint(6, len(SLOTS))))
    current_value = {}
    for session_id in range(n_sessions):
        if not active_slots:
            break
        # 1-3 turns per session
        n_turns = rng.randint(1, 3)
        session_slots = rng.sample(active_slots, k=min(n_turns, len(active_slots)))
        for slot in session_slots:
            turn_counter += 1
            if slot in current_value and rng.random() < 0.12:
                # deletion request: permanent for this user and slot (matches a
                # real deletable-memory product, where "forget X" does not leave
                # the door open for X to quietly reappear later in the same
                # timeline), so the slot is removed from the active pool entirely
                tl.turns.append(Turn(session_id, turn_counter, slot, None, True, DELETE_TEMPLATE.format(slot_label=SLOT_LABEL[slot])))
                tl.deleted_slots.add(slot)
                current_value.pop(slot, None)
                active_slots.remove(slot)
                continue
            # new value: either first mention, or (rarely) an update/contradiction
            pool = [v for v in VALUE_POOLS[slot] if v != current_value.get(slot)]
            value = rng.choice(pool) if pool else VALUE_POOLS[slot][0]
            current_value[slot] = value
            tl.deleted_slots.discard(slot)
            tl.turns.append(Turn(session_id, turn_counter, slot, value, False, TEMPLATE_BY_SLOT[slot].format(value=value)))
    tl.ground_truth_current = dict(current_value)
    return tl


def generate_corpus(n_users: int, n_sessions_per_user: int, seed: int) -> list[UserTimeline]:
    rng = random.Random(seed)
    return [generate_user_timeline(f"user_{i:04d}", n_sessions_per_user, random.Random(seed * 100003 + i)) for i in range(n_users)]
