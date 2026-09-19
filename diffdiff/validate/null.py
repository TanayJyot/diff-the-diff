"""The null test: diff a model against itself.

Every feature a diffing method reports as model-exclusive when the two inputs
are *identical* is, by construction, a false positive. That makes this the
cheapest possible falsification of the method, and it establishes a
false-discovery floor that every downstream finding must be discounted by.

arXiv:2602.11729 does not report this experiment. It matters most for a DFC,
whose exclusive partitions are a fixed fraction of the dictionary allocated up
front: that capacity exists whether or not anything exclusive does, and the
reconstruction objective has every incentive to use it.

The headline numbers are:

``exclusive_mass_frac``
    Share of total feature activation mass routed through partitions that
    should be empty. A method that correctly finds nothing drives this to zero.

``mirror_excess``
    How much more often exclusive features have a near-duplicate in the
    opposing partition than shared features do by chance.

``standard_extreme_frac``
    For a standard crosscoder, the fraction of features whose relative decoder
    norm is extreme enough that a practitioner would report them as exclusive.
    Also false positives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from diffdiff.data.toy import ToyActivations, ToyConfig, null_config
from diffdiff.diffing.crosscoder import Crosscoder, CrosscoderConfig
from diffdiff.diffing.train import (
    ActivationNormalizer,
    TrainConfig,
    collect_features,
    train,
)
from diffdiff.validate.mirror import MirrorReport, find_mirror_pairs


@dataclass
class NullReport:
    """Outcome of a null test.

    Attributes:
        exclusive_live_frac: Fraction of exclusive-partition features that fire
            at least once on held-out data.
        exclusive_mass_frac: Share of total activation mass carried by exclusive
            partitions. The headline number.
        expected_mass_frac: Mass share the exclusive partitions would carry if
            activation were spread uniformly across the dictionary — the
            reference point for "no preference", not for "correct".
        mirror: Mirror-pair analysis of the exclusive partitions.
        fve_a: Reconstruction quality for model A, to confirm training worked.
        fve_b: Reconstruction quality for model B.
        l0: Mean active features per sample.
        standard_extreme_frac: Fraction of a *standard* crosscoder's features
            with relative decoder norm outside ``[0.1, 0.9]``, i.e. what a
            practitioner would report as exclusive. ``None`` if not run.
        seed: Seed for this run.
        history: Training metric history for the DFC.
    """

    exclusive_live_frac: float
    exclusive_mass_frac: float
    expected_mass_frac: float
    mirror: MirrorReport
    fve_a: float
    fve_b: float
    l0: float
    standard_extreme_frac: float | None = None
    seed: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    def verdict(self) -> str:
        """A one-line reading of whether the regime is usable."""
        if self.exclusive_mass_frac < 0.01 and self.mirror.excess < 0.05:
            return "PASS - exclusive partitions stayed essentially empty"
        if self.mirror.excess >= 0.2:
            return "FAIL - partitions filled with mirror pairs (manufactured differences)"
        if self.exclusive_mass_frac >= 0.05:
            return "FAIL - substantial activation mass routed through empty-by-construction partitions"
        return "MARGINAL - non-trivial occupancy without clear mirroring; inspect features"

    def summary(self) -> str:
        std = (
            "n/a"
            if self.standard_extreme_frac is None
            else f"{self.standard_extreme_frac:.3f}"
        )
        return (
            f"seed={self.seed} | {self.verdict()}\n"
            f"  exclusive mass {self.exclusive_mass_frac:.4f} "
            f"(uniform would be {self.expected_mass_frac:.4f}), "
            f"live {self.exclusive_live_frac:.3f}\n"
            f"  {self.mirror.summary()}\n"
            f"  standard-crosscoder extreme-norm frac {std}\n"
            f"  FVE a/b {self.fve_a:.3f}/{self.fve_b:.3f}, L0 {self.l0:.1f}"
        )


@torch.no_grad()
def _partition_stats(
    model: Crosscoder, features: torch.Tensor
) -> tuple[float, float]:
    """Live fraction and activation-mass share of the exclusive partitions."""
    excl = (model.partition_mask("A") | model.partition_mask("B")).cpu()
    live = (features > 0).any(dim=0)
    live_frac = float((live & excl).sum() / max(1, int(excl.sum())))

    mass = features.sum(dim=0)
    total = mass.sum().clamp_min(1e-8)
    return live_frac, float(mass[excl].sum() / total)


def run_regime_test(
    toy: ToyConfig,
    crosscoder: CrosscoderConfig | None = None,
    training: TrainConfig | None = None,
    eval_batches: int = 20,
    compare_standard: bool = True,
    mirror_threshold: float = 0.5,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> NullReport:
    """Train a DFC on a toy regime and measure its exclusive partitions.

    Shared by the null test and the positive control, so the two are measured by
    identical code. :func:`run_null_test` wraps this with the null guard.

    Args:
        toy: Activation source.
        crosscoder: DFC configuration. Defaults to a 5% partition, matching the
            paper's main setting.
        training: Training configuration.
        eval_batches: Held-out batches used for the feature statistics.
        compare_standard: Also train a standard crosscoder for reference.
        mirror_threshold: Correlation threshold for mirror pairs.
        seed: Seed applied to the toy data, the model, and training.
        progress: Optional callback for human-readable progress lines.

    Returns:
        A :class:`NullReport`.
    """
    crosscoder = crosscoder or CrosscoderConfig(
        d_a=toy.d_a, d_b=toy.d_b, n_features=4096, exclusive_frac=0.05, k=16, k_initial=64
    )
    training = training or TrainConfig(seed=seed)

    source = ToyActivations(toy)
    if toy.is_null:
        x_a, x_b = source.sample_activations(64)
        if not torch.equal(x_a, x_b):
            raise ValueError("null configuration did not produce identical activations")

    def batch_fn(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return source.sample_activations(n)

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    torch.manual_seed(seed)
    say(f"training DFC ({crosscoder.n_exclusive} features per exclusive partition)")
    dfc = Crosscoder(crosscoder)
    history = train(
        dfc, batch_fn, training,
        progress=lambda s, m: say(
            f"  step {s:>5} loss {m['loss']:.4f} fve {m['fve_a']:.3f} "
            f"l0 {m['l0']:.1f} dead {m['dead_frac']:.3f}"
        ),
    )

    say("collecting held-out features")
    cal_a, cal_b = batch_fn(min(4096, training.batch_size * 8))
    normalizer = ActivationNormalizer(cal_a, cal_b)
    features = collect_features(
        dfc, batch_fn, eval_batches, training.batch_size,
        device=training.device, normalizer=normalizer,
    )

    live_frac, mass_frac = _partition_stats(dfc, features)
    mirror = find_mirror_pairs(dfc, features, threshold=mirror_threshold, seed=seed)

    standard_extreme: float | None = None
    if compare_standard:
        say("training standard crosscoder for reference")
        std_cfg = CrosscoderConfig(
            **{**crosscoder.__dict__, "exclusive_frac": 0.0}
        )
        torch.manual_seed(seed)
        std = Crosscoder(std_cfg)
        train(std, batch_fn, training)
        rel = std.relative_decoder_norm()
        standard_extreme = float(((rel > 0.9) | (rel < 0.1)).float().mean())

    final = history[-1] if history else {"fve_a": 0.0, "fve_b": 0.0, "l0": 0.0}
    n_excl = 2 * crosscoder.n_exclusive
    return NullReport(
        exclusive_live_frac=live_frac,
        exclusive_mass_frac=mass_frac,
        expected_mass_frac=n_excl / crosscoder.n_features,
        mirror=mirror,
        fve_a=final["fve_a"],
        fve_b=final["fve_b"],
        l0=final["l0"],
        standard_extreme_frac=standard_extreme,
        seed=seed,
        history=history,
    )


def run_null_test(
    toy: ToyConfig | None = None,
    **kwargs,
) -> NullReport:
    """Run :func:`run_regime_test` on a configuration guaranteed to be a null.

    Raises:
        ValueError: If ``toy`` does not describe a true null, since the whole
            point of this test is that any exclusive feature it finds is known
            in advance to be a false positive.
    """
    toy = toy or null_config(seed=kwargs.get("seed", 0))
    if not toy.is_null:
        raise ValueError(
            "run_null_test requires a null configuration (identical inputs); "
            "got exclusive concepts, a non-identity transform, or added noise"
        )
    return run_regime_test(toy, **kwargs)


def run_control_test(toy: ToyConfig, **kwargs) -> NullReport:
    """Positive control: a regime where exclusive concepts genuinely exist.

    A null test only means something alongside this. If the exclusive partitions
    stay empty here too, the method is not finding differences at this scale and
    a clean null is vacuous rather than reassuring.

    Raises:
        ValueError: If ``toy`` has no exclusive concepts to find.
    """
    if toy.n_excl_a == 0 and toy.n_excl_b == 0 and not toy.drop_concepts:
        raise ValueError(
            "a positive control needs exclusive or dropped concepts to recover"
        )
    return run_regime_test(toy, **kwargs)
