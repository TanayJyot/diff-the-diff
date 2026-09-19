"""Aligning activations between two models.

For quantization and merge verification both models share a tokenizer, so
alignment is the identity and :class:`IdentityAligner` applies. The interface
exists from day one anyway so that cross-architecture diffing is a new
implementation rather than a rewrite — see ``docs/DESIGN.md`` §5.

:class:`WindowExpansionAligner` implements Algorithm 1 of arXiv:2602.11729 for
the cross-tokenizer case: walk both token sequences, try a decoded one-to-one
match, and on mismatch grow the window on whichever side has the shorter decoded
text until the decoded strings agree. On a match, keep only the *final* token's
activation from each window.

That last-token rule is the load-bearing assumption of the whole approach — the
paper's stated hope is that "the final token's activation captures the semantic
context of the entire window" — and it is not validated there. It is cheap for
us to probe once we have two real tokenizers, and it is the most likely place
for cross-architecture diffing to degrade quietly.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Protocol, Sequence


class Aligner(Protocol):
    """Maps two token sequences to index pairs referring to the same content."""

    def align(self, tokens_a: Sequence[int], tokens_b: Sequence[int]) -> list[tuple[int, int]]:
        ...


def normalize(text: str) -> str:
    """Normalise decoded text before comparison.

    NFKC folding plus whitespace stripping absorbs the differences that make two
    tokenizers disagree about identical content. The paper reports its residual
    failures concentrate in smart quotes, box-drawing characters and
    multi-codepoint emoji, which is what this addresses.
    """
    return unicodedata.normalize("NFKC", text).strip()


class IdentityAligner:
    """For models that share a tokenizer: position ``i`` maps to position ``i``."""

    def align(self, tokens_a: Sequence[int], tokens_b: Sequence[int]) -> list[tuple[int, int]]:
        if len(tokens_a) != len(tokens_b):
            raise ValueError(
                f"identity alignment needs equal lengths, got {len(tokens_a)} and {len(tokens_b)}"
            )
        return [(i, i) for i in range(len(tokens_a))]


class WindowExpansionAligner:
    """Greedy decoded-text alignment across different tokenizers (Algorithm 1).

    Args:
        tokenizer_a: Tokenizer for model A; needs ``decode``.
        tokenizer_b: Tokenizer for model B.
        max_window: Largest window to expand to before giving up on a position.
        skip_non_content: Skip whitespace-only and special tokens.
    """

    def __init__(
        self,
        tokenizer_a: Any,
        tokenizer_b: Any,
        max_window: int = 16,
        skip_non_content: bool = True,
    ):
        self.tok_a = tokenizer_a
        self.tok_b = tokenizer_b
        self.max_window = max_window
        self.skip_non_content = skip_non_content
        self._special_a = set(getattr(tokenizer_a, "all_special_ids", []) or [])
        self._special_b = set(getattr(tokenizer_b, "all_special_ids", []) or [])

    def _is_non_content(self, token: int, tokenizer: Any, special: set[int]) -> bool:
        if token in special:
            return True
        return normalize(tokenizer.decode([token])) == ""

    def align(
        self, tokens_a: Sequence[int], tokens_b: Sequence[int]
    ) -> list[tuple[int, int]]:
        """Return ``(index_a, index_b)`` pairs of semantically matched positions.

        Each pair is the *final* token of a matched window, giving a strict 1:1
        activation mapping. Positions that cannot be matched are dropped.
        """
        pairs: list[tuple[int, int]] = []
        pa = pb = 0
        na, nb = len(tokens_a), len(tokens_b)

        while pa < na and pb < nb:
            if self.skip_non_content:
                while pa < na and self._is_non_content(tokens_a[pa], self.tok_a, self._special_a):
                    pa += 1
                while pb < nb and self._is_non_content(tokens_b[pb], self.tok_b, self._special_b):
                    pb += 1
                if pa >= na or pb >= nb:
                    break

            sa = normalize(self.tok_a.decode(list(tokens_a[pa : pa + 1])))
            sb = normalize(self.tok_b.decode(list(tokens_b[pb : pb + 1])))

            if sa == sb:
                pairs.append((pa, pb))
                pa += 1
                pb += 1
                continue

            matched = self._expand(tokens_a, tokens_b, pa, pb)
            if matched is None:
                # Unresolvable: advance both and resynchronise rather than
                # abandoning the rest of the sequence.
                pa += 1
                pb += 1
                continue
            ea, eb = matched
            pairs.append((ea - 1, eb - 1))  # final token of each window
            pa, pb = ea, eb

        return pairs

    def _expand(
        self, tokens_a: Sequence[int], tokens_b: Sequence[int], pa: int, pb: int
    ) -> tuple[int, int] | None:
        """Grow windows until the decoded text agrees; return the two end indices."""
        na, nb = len(tokens_a), len(tokens_b)
        ea, eb = pa + 1, pb + 1

        while (ea - pa) <= self.max_window and (eb - pb) <= self.max_window:
            wa = normalize(self.tok_a.decode(list(tokens_a[pa:ea])))
            wb = normalize(self.tok_b.decode(list(tokens_b[pb:eb])))
            if wa == wb:
                return ea, eb
            # Grow the side whose decoded text is shorter — the asymmetric step
            # that lets ['198','9'] catch up to ['1989'].
            if len(wa) < len(wb) and ea < na:
                ea += 1
            elif eb < nb:
                eb += 1
            elif ea < na:
                ea += 1
            else:
                return None
        return None


def alignment_rate(pairs: list[tuple[int, int]], n_tokens: int) -> float:
    """Fraction of positions successfully aligned.

    The paper reports >99% on FineWeb/LMSYS text. A materially lower rate on our
    own corpus means the corpus, not the method, needs attention.
    """
    return len(pairs) / max(1, n_tokens)
