from src.dialogue_gen import QUERY_TEMPLATE_BY_SLOT, SLOTS, TEMPLATE_BY_SLOT, VALUE_POOLS
from src.embedding import train_embedder


def test_training_loss_decreases():
    _model, losses = train_embedder(epochs=30)
    assert losses[-1] < losses[0]


def test_question_phrasing_nearest_neighbor_matches_its_own_slot(embedder):
    import torch

    correct = 0
    for slot in SLOTS:
        q = torch.tensor(embedder.embed_text(QUERY_TEMPLATE_BY_SLOT[slot]))
        best_slot, best_sim = None, -2.0
        for other_slot in SLOTS:
            for value in VALUE_POOLS[other_slot][:2]:
                e = torch.tensor(embedder.embed_text(TEMPLATE_BY_SLOT[other_slot].format(value=value)))
                sim = float((q * e).sum())
                if sim > best_sim:
                    best_sim, best_slot = sim, other_slot
        correct += int(best_slot == slot)
    assert correct == len(SLOTS)
