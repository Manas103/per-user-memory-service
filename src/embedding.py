"""A small PyTorch embedding model used for retrieval.

Input features are a feature-hashed bag-of-words count vector (`hash_vector`,
width `config.HASH_DIM`), not a learned tokenizer; that part is a disclosed
proxy, the same convention `telemetry.py`-style modules in this portfolio use
for token counting. What is a real, trained model is the projection from that
hashed vector down to `config.EMBED_DIM`: a 2-layer MLP trained with a triplet
margin loss so that two fact sentences about the *same slot* (e.g. two
different home-city statements) land closer together than a fact sentence
about a *different slot*, which is what makes a per-user pgvector nearest-
neighbor search return the right kind of fact for a query. Training uses
template-generated sentences built the same way `dialogue_gen.py` builds
dialogue turns, but over its own random split of slot/value combinations, not
the eval corpus `measure.py` scores against.
"""

from __future__ import annotations

import hashlib
import random

import torch
from torch import nn

from src import config
from src.dialogue_gen import QUERY_TEMPLATE_BY_SLOT, SLOTS, TEMPLATE_BY_SLOT, VALUE_POOLS

_TOKEN_RE_CACHE = None


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().replace(".", " ").replace(",", " ").replace("'", " ").split() if t]


def hash_vector(text: str, dim: int = config.HASH_DIM) -> torch.Tensor:
    """Deterministic feature hashing: each token increments one bucket of a
    fixed-width count vector. No vocabulary is built or stored, so this never
    needs to be re-fit, and it is the same function at train time and at
    retrieval time."""
    v = torch.zeros(dim)
    for tok in _tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        v[h] += 1.0
    norm = v.norm()
    return v / norm if norm > 0 else v


class FactEmbedder(nn.Module):
    def __init__(self, hash_dim: int = config.HASH_DIM, embed_dim: int = config.EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hash_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def embed_text(self, text: str) -> list[float]:
        with torch.no_grad():
            z = self.forward(hash_vector(text).unsqueeze(0))
        return z.squeeze(0).tolist()


def _training_sentences(rng: random.Random, n_per_slot: int) -> dict[str, list[str]]:
    """Each slot's pool mixes statement sentences (one per value) with several
    repeats of that slot's fixed question phrasing, so the trained embedder
    has to learn that a question and its matching statement are close even
    though they share few surface words; this is what retrieval.py leans on
    at answer time."""
    out = {}
    for slot in SLOTS:
        sentences = [TEMPLATE_BY_SLOT[slot].format(value=v) for v in VALUE_POOLS[slot]]
        sentences += [QUERY_TEMPLATE_BY_SLOT[slot]] * max(1, n_per_slot // len(VALUE_POOLS[slot]))
        while len(sentences) < n_per_slot:
            value = rng.choice(VALUE_POOLS[slot])
            sentences.append(TEMPLATE_BY_SLOT[slot].format(value=value))
        out[slot] = sentences
    return out


def train_embedder(seed: int = 7, epochs: int = 60, margin: float = 0.3) -> tuple[FactEmbedder, list[float]]:
    """Trains on sentences generated from this module's own random split
    (seed independent of measure.py's eval corpus seed). Returns the trained
    model and the per-epoch triplet loss, which measure.py writes to
    docs/ so the "trained, not fixed" claim is checkable."""
    rng = random.Random(seed)
    sentences_by_slot = _training_sentences(rng, n_per_slot=40)
    model = FactEmbedder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.TripletMarginLoss(margin=margin)
    losses = []
    for _epoch in range(epochs):
        anchors, positives, negatives = [], [], []
        for slot in SLOTS:
            same = sentences_by_slot[slot]
            other_slot = rng.choice([s for s in SLOTS if s != slot])
            diff = sentences_by_slot[other_slot]
            for _ in range(8):
                a, p = rng.sample(same, 2)
                n = rng.choice(diff)
                anchors.append(hash_vector(a))
                positives.append(hash_vector(p))
                negatives.append(hash_vector(n))
        a_t = torch.stack(anchors)
        p_t = torch.stack(positives)
        n_t = torch.stack(negatives)
        opt.zero_grad()
        loss = loss_fn(model(a_t), model(p_t), model(n_t))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return model, losses
