"""Ground-truth concept recovery for cross-architecture diffing.

Phase 0 measured what a method invents when there is nothing to find. This
measures the other half: when exclusive concepts genuinely exist, does the
method find *those* concepts, or merely populate its exclusive partition?

The toy model knows which concepts are exclusive and what direction each one
occupies, so both questions are answerable exactly rather than by proxy. This is
the setting of arXiv:2602.11729 Figure 4, with two changes:

1. **The exclusive budget is matched across architectures.** A DFC labels its
   dedicated partition exclusive; a standard crosscoder has no partition, so we
   take the same number of most-extreme relative-decoder-norm features. Without
   this, a method that simply labels more features exclusive scores higher on
   recall for free.

2. **Precision is reported alongside recall.** Phase 0 showed that forced
   allocation inflates recall on its own: a method that fills its exclusive
   partition regardless will recover more exclusive concepts *and* more
   non-exclusive ones. Recall without precision cannot distinguish discovery
   from indiscriminate labelling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffdiff.data.toy import ToyActivations
from diffdiff.diffing.crosscoder import Crosscoder


@dataclass
class RecoveryReport:
    """Concept-level precision and recall for one model's exclusive partition.

    Attributes:
        model: ``"a"`` or ``"b"``.
        recall: Fraction of genuinely exclusive concepts matched by a feature
            the method labels exclusive.
        precision: Fraction of features labelled exclusive that match a
            genuinely exclusive concept.
        f1: Harmonic mean of the two.
        shared_leakage: Fraction of features labelled exclusive that actually
            match a *shared* concept. These are the false positives that matter
            most, because a shared concept mislabelled as exclusive is exactly
            the "difference" a user would investigate and find spurious.
        unmatched: Fraction of features labelled exclusive that match no
            ground-truth concept at all above threshold.
        n_true: Number of genuinely exclusive concepts.
        n_labelled: Number of features labelled exclusive (budget-matched).
        threshold: Cosine similarity required to call a feature-concept match.
        best_cos_median: Median best-match cosine among labelled features, a
            read on whether matches are crisp or marginal.
    """

    model: str
    recall: float
    precision: float
    f1: float
    shared_leakage: float
    unmatched: float
    n_true: int
    n_labelled: int
    threshold: float
    best_cos_median: float

    def summary(self) -> str:
        return (
            f"[{self.model}] recall={self.recall:.3f} precision={self.precision:.3f} "
            f"f1={self.f1:.3f} | shared_leakage={self.shared_leakage:.3f} "
            f"unmatched={self.unmatched:.3f} | true={self.n_true} "
            f"labelled={self.n_labelled} med_cos={self.best_cos_median:.3f}"
        )


def _cosine_matrix(features: torch.Tensor, concepts: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between every feature direction and every concept.

    Returns ``(n_features, n_concepts)``. Rows for features with a zero decoder
    (outside this model's partition) come back as zero rather than NaN.
    """
    f = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c = concepts / concepts.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return f @ c.T


def concept_recovery(
    model: Crosscoder,
    source: ToyActivations,
    which: str = "a",
    threshold: float = 0.5,
) -> RecoveryReport:
    """Score a trained crosscoder's exclusive partition against ground truth.

    Args:
        model: A trained crosscoder or DFC.
        source: The toy generator that produced its training data, carrying the
            ground-truth concept directions and visibility masks.
        which: Which model's exclusive partition to score, ``"a"`` or ``"b"``.
        threshold: Cosine similarity required to count a feature as having
            recovered a concept.

    Returns:
        A :class:`RecoveryReport`.
    """
    if which not in ("a", "b"):
        raise ValueError(f"which must be 'a' or 'b', got {which!r}")

    gt = source.ground_truth()
    concepts = source.concepts_a if which == "a" else source.concepts_b
    # A concept is genuinely exclusive to this model when it is visible here and
    # not in the other model. This covers both planted exclusives and concepts
    # destroyed on the other side.
    visible_here = gt["visible_a"] if which == "a" else gt["visible_b"]
    visible_other = gt["visible_b"] if which == "a" else gt["visible_a"]
    true_exclusive = visible_here & ~visible_other
    truly_shared = visible_here & visible_other

    decoders = model.full_decoder(which)
    labelled = model.exclusive_mask(which).cpu()

    cos = _cosine_matrix(decoders, concepts)
    best_cos, best_concept = cos.max(dim=-1)

    n_true = int(true_exclusive.sum())
    n_labelled = int(labelled.sum())
    if n_labelled == 0:
        return RecoveryReport(which, 0.0, 0.0, 0.0, 0.0, 0.0, n_true, 0, threshold, 0.0)

    # Precision side: what did the labelled features turn out to be?
    lab_idx = labelled.nonzero(as_tuple=True)[0]
    lab_cos = best_cos[lab_idx]
    lab_concept = best_concept[lab_idx]
    matched = lab_cos >= threshold

    hits = matched & true_exclusive[lab_concept]
    leaks = matched & truly_shared[lab_concept]
    precision = float(hits.float().mean())
    shared_leakage = float(leaks.float().mean())
    unmatched = float((~matched).float().mean())

    # Recall side: which true exclusive concepts were picked up by a labelled
    # feature? Scored per concept, so one feature cannot claim several.
    if n_true > 0:
        true_idx = true_exclusive.nonzero(as_tuple=True)[0]
        cos_labelled = cos[lab_idx][:, true_idx]          # (n_labelled, n_true)
        recovered = (cos_labelled >= threshold).any(dim=0)
        recall = float(recovered.float().mean())
    else:
        recall = 0.0

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return RecoveryReport(
        model=which,
        recall=recall,
        precision=precision,
        f1=f1,
        shared_leakage=shared_leakage,
        unmatched=unmatched,
        n_true=n_true,
        n_labelled=n_labelled,
        threshold=threshold,
        best_cos_median=float(lab_cos.median()),
    )
