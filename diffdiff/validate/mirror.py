"""Mirror-feature detection.

A *mirror pair* is a feature in ``I_A`` and a feature in ``I_B`` that encode the
same underlying concept. Because a DFC allocates a fixed fraction of its
dictionary to each exclusive partition whether or not anything exclusive exists,
mirror pairs are the expected failure mode when the two models are nearly
identical — the partition has capacity to fill, so it fills with duplicates of
shared structure. Those duplicates look exactly like discovered differences.

arXiv:2602.11729 §5.2 reports encountering this on base-vs-finetune pairs and
leaves it as future work. Quantization is a more extreme version of the same
regime, so we need to measure it directly.

Two signals, with different requirements:

**Co-firing correlation** (primary). Two features that encode the same concept
fire on the same inputs. This is computed from activations alone, so it works
regardless of whether the two models share a representational space — it is the
signal that will still apply once we move to cross-architecture diffing.

**Decoder cosine** (secondary). Only meaningful when the two decoder spaces are
comparable: the same model diffed against itself, or a quantized pair. For
genuinely different architectures this requires a stitching map first, so it is
reported as ``None`` when the dimensions differ.

Both are scored against a **control**: the same statistic computed between two
random halves of the *shared* partition. Taking a maximum over many candidates
inflates the score on its own, so an uncontrolled mirror rate is not
interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffdiff.diffing.crosscoder import Crosscoder


@dataclass
class MirrorReport:
    """Result of a mirror-pair search.

    Attributes:
        mirror_frac: Fraction of live A-exclusive features whose best
            B-exclusive partner exceeds ``threshold``.
        control_frac: The same fraction computed between two random halves of
            the shared partition — what this statistic looks like by chance.
        excess: ``mirror_frac - control_frac``. The number that matters.
        median_score: Median best-partner correlation among live A-exclusive
            features.
        control_median: Median best-partner correlation in the control.
        median_decoder_cos: Median decoder cosine among matched pairs, or
            ``None`` when the two decoder spaces are not comparable.
        n_live_a: Live features in ``I_A``.
        n_live_b: Live features in ``I_B``.
        threshold: Correlation threshold used.
        pairs: ``(a_index, b_index, score)`` for pairs above threshold, sorted
            by descending score.
    """

    mirror_frac: float
    control_frac: float
    excess: float
    median_score: float
    control_median: float
    median_decoder_cos: float | None
    n_live_a: int
    n_live_b: int
    threshold: float
    pairs: list[tuple[int, int, float]]

    def summary(self) -> str:
        cos = "n/a" if self.median_decoder_cos is None else f"{self.median_decoder_cos:.3f}"
        return (
            f"mirror_frac={self.mirror_frac:.3f} control={self.control_frac:.3f} "
            f"excess={self.excess:+.3f} median_corr={self.median_score:.3f} "
            f"(control {self.control_median:.3f}) decoder_cos={cos} "
            f"live A/B={self.n_live_a}/{self.n_live_b}"
        )


def _standardize(features: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean, unit-variance each feature column; flag degenerate ones."""
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    std = centered.pow(2).mean(dim=0, keepdim=True).sqrt()
    live = (std > eps).squeeze(0)
    return centered / std.clamp_min(eps), live


def _best_partner(
    left: torch.Tensor, right: torch.Tensor, chunk: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max correlation and argmax for each column of ``left`` against ``right``.

    Chunked over ``left`` so the full cross-correlation matrix is never
    materialised; with a 5% partition of a 131k dictionary that matrix would be
    43M entries.
    """
    n_samples = left.shape[0]
    scores, indices = [], []
    for start in range(0, left.shape[1], chunk):
        block = left[:, start : start + chunk]
        corr = (block.T @ right) / n_samples
        best, idx = corr.max(dim=-1)
        scores.append(best)
        indices.append(idx)
    return torch.cat(scores), torch.cat(indices)


def find_mirror_pairs(
    model: Crosscoder,
    features: torch.Tensor,
    threshold: float = 0.5,
    seed: int = 0,
) -> MirrorReport:
    """Search a trained DFC's exclusive partitions for mirrored concepts.

    Args:
        model: A trained DFC. Calling this on a standard crosscoder (no
            dedicated partitions) raises ``ValueError``.
        features: Feature activations over a held-out sample, shape
            ``(n_samples, n_features)``.
        threshold: Correlation above which a pair counts as mirrored.
        seed: Seed for the control's random split.

    Returns:
        A :class:`MirrorReport`.
    """
    if not model.config.is_dfc:
        raise ValueError("mirror analysis requires dedicated partitions (exclusive_frac > 0)")

    std_feats, live = _standardize(features)
    mask_a = model.partition_mask("A").cpu()
    mask_b = model.partition_mask("B").cpu()
    mask_s = model.partition_mask("S").cpu()

    live_a = (mask_a & live).nonzero(as_tuple=True)[0]
    live_b = (mask_b & live).nonzero(as_tuple=True)[0]
    live_s = (mask_s & live).nonzero(as_tuple=True)[0]

    if len(live_a) == 0 or len(live_b) == 0:
        # A partition that never fires is itself the answer: nothing was
        # manufactured, because nothing is there.
        return MirrorReport(
            mirror_frac=0.0, control_frac=0.0, excess=0.0,
            median_score=0.0, control_median=0.0, median_decoder_cos=None,
            n_live_a=len(live_a), n_live_b=len(live_b),
            threshold=threshold, pairs=[],
        )

    scores, partners = _best_partner(std_feats[:, live_a], std_feats[:, live_b])

    # Control: two random halves of the shared partition. Shared features are
    # not duplicates of each other, so this measures how large a maximum over
    # this many candidates gets by chance.
    gen = torch.Generator().manual_seed(seed)
    perm = live_s[torch.randperm(len(live_s), generator=gen)]
    half = min(len(live_a), len(perm) // 2)
    if half > 0:
        ctrl_l, ctrl_r = perm[:half], perm[half : half + max(half, len(live_b))]
        ctrl_scores, _ = _best_partner(std_feats[:, ctrl_l], std_feats[:, ctrl_r])
    else:
        ctrl_scores = torch.zeros(1)

    above = scores >= threshold
    mirror_frac = float(above.float().mean())
    control_frac = float((ctrl_scores >= threshold).float().mean())

    # Decoder cosine is only meaningful when both decoders live in comparable
    # spaces — the same model against itself, or a quantized pair.
    median_cos: float | None = None
    if model.config.d_a == model.config.d_b and above.any():
        dec_a = model.W_dec_a.detach().cpu()
        dec_b = model.W_dec_b.detach().cpu()
        off_a, off_b = model.slice_a.start, model.slice_b.start
        cos = []
        for i, j in zip(live_a[above].tolist(), live_b[partners[above]].tolist()):
            va = dec_a[i - off_a]
            vb = dec_b[j - off_b]
            cos.append(float(torch.nn.functional.cosine_similarity(va, vb, dim=0)))
        median_cos = float(torch.tensor(cos).median())

    pairs = sorted(
        (
            (int(a), int(live_b[p]), float(s))
            for a, p, s in zip(live_a[above], partners[above], scores[above])
        ),
        key=lambda t: -t[2],
    )

    return MirrorReport(
        mirror_frac=mirror_frac,
        control_frac=control_frac,
        excess=mirror_frac - control_frac,
        median_score=float(scores.median()),
        control_median=float(ctrl_scores.median()),
        median_decoder_cos=median_cos,
        n_live_a=len(live_a),
        n_live_b=len(live_b),
        threshold=threshold,
        pairs=pairs,
    )
