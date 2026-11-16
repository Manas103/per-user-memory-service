from src.dialogue_gen import generate_corpus
from src.extractor import ExtractedDeletion, ExtractedFact, extract


def test_extract_statement_inverts_template_exactly():
    corpus = generate_corpus(n_users=5, n_sessions_per_user=6, seed=11)
    checked = 0
    for tl in corpus:
        for turn in tl.turns:
            result = extract(turn.text)
            if turn.is_deletion_request:
                assert isinstance(result, ExtractedDeletion)
                assert result.slot == turn.slot
            else:
                assert isinstance(result, ExtractedFact)
                assert result.slot == turn.slot
                assert result.value == turn.value
            checked += 1
    assert checked > 50


def test_extract_returns_none_for_unknown_text():
    assert extract("This sentence matches no slot template at all.") is None
