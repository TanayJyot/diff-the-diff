"""Aligner tests using fake tokenizers, so no model weights are needed."""

import pytest

from diffdiff.align.aligner import (
    IdentityAligner,
    WindowExpansionAligner,
    alignment_rate,
    normalize,
)


class FakeTokenizer:
    """Maps integer ids to strings via a vocabulary list."""

    def __init__(self, vocab: list[str], special_ids: list[int] | None = None):
        self.vocab = vocab
        self.all_special_ids = special_ids or []

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.vocab[i] for i in ids)

    def ids(self, pieces: list[str]) -> list[int]:
        return [self.vocab.index(p) for p in pieces]


def test_normalize_folds_and_strips():
    assert normalize("  hello  ") == "hello"
    assert normalize("ﬁ") == "fi"  # NFKC ligature folding


def test_identity_aligner():
    pairs = IdentityAligner().align([1, 2, 3], [4, 5, 6])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_identity_aligner_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal lengths"):
        IdentityAligner().align([1, 2], [1, 2, 3])


def test_window_expansion_matches_the_papers_1989_example():
    """Model A has ['1989']; model B splits it into ['198','9']."""
    vocab_a = ["In", " ", "1989", " and"]
    vocab_b = ["In", " ", "198", "9", " and"]
    tok_a = FakeTokenizer(vocab_a)
    tok_b = FakeTokenizer(vocab_b)

    tokens_a = tok_a.ids(["In", "1989", " and"])
    tokens_b = tok_b.ids(["In", "198", "9", " and"])

    aligner = WindowExpansionAligner(tok_a, tok_b)
    pairs = aligner.align(tokens_a, tokens_b)

    assert (0, 0) in pairs, "the leading token should match one-to-one"
    # '1989' in A aligns to the FINAL token of B's window, i.e. '9' at index 2.
    assert (1, 2) in pairs
    # And alignment resynchronises afterwards.
    assert (2, 3) in pairs


def test_window_expansion_keeps_final_token_of_each_window():
    """Many-to-many windows still yield exactly one pair, at the window ends."""
    tok_a = FakeTokenizer(["ab", "cd"])
    tok_b = FakeTokenizer(["a", "b", "c", "d"])
    pairs = WindowExpansionAligner(tok_a, tok_b).align([0, 1], [0, 1, 2, 3])
    assert pairs == [(0, 1), (1, 3)]


def test_window_expansion_identical_tokenizers_is_one_to_one():
    tok = FakeTokenizer(["hello", " ", "world"])
    pairs = WindowExpansionAligner(tok, tok).align([0, 2], [0, 2])
    assert pairs == [(0, 0), (1, 1)]


def test_window_expansion_skips_special_tokens():
    vocab = ["<bos>", "hi", "there"]
    tok_a = FakeTokenizer(vocab, special_ids=[0])
    tok_b = FakeTokenizer(vocab, special_ids=[0])
    pairs = WindowExpansionAligner(tok_a, tok_b).align([0, 1, 2], [1, 2])
    assert pairs == [(1, 0), (2, 1)]


def test_window_expansion_gives_up_past_max_window():
    """Unmatchable content is dropped, not allowed to desynchronise the rest."""
    tok_a = FakeTokenizer(["xxxx", "tail"])
    tok_b = FakeTokenizer(["y", "tail"])
    pairs = WindowExpansionAligner(tok_a, tok_b, max_window=2).align([0, 1], [0, 1])
    # 'xxxx' vs 'y' cannot match, but the aligner resynchronises on 'tail'.
    assert (1, 1) in pairs


def test_alignment_rate():
    assert alignment_rate([(0, 0), (1, 1)], 4) == 0.5
    assert alignment_rate([], 0) == 0.0
