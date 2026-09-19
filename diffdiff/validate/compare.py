"""Head-to-head comparison of crosscoder architectures on a cross-architecture diff.

Trains a DFC and a standard crosscoder on the same data with the same budget,
and scores both against ground truth. This is the paper's Figure 4 comparison,
with the exclusive budget matched and precision reported alongside recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from diffdiff.data.toy import ToyActivations, ToyConfig
from diffdiff.diffing.crosscoder import Crosscoder, CrosscoderConfig
from diffdiff.diffing.train import TrainConfig, train
from diffdiff.validate.recovery import RecoveryReport, concept_recovery


@dataclass
class ComparisonReport:
    """Recovery scores for one architecture on one seed.

    Attributes:
        architecture: ``"dfc"`` or ``"standard"``.
        report_a: Recovery for model A's exclusive partition.
        report_b: Recovery for model B's exclusive partition.
        fve_a: Final reconstruction quality for A.
        fve_b: Final reconstruction quality for B.
        seed: Seed for this run.
    """

    architecture: str
    report_a: RecoveryReport
    report_b: RecoveryReport
    fve_a: float
    fve_b: float
    seed: int

    @property
    def mean_recall(self) -> float:
        return 0.5 * (self.report_a.recall + self.report_b.recall)

    @property
    def mean_precision(self) -> float:
        return 0.5 * (self.report_a.precision + self.report_b.precision)

    @property
    def mean_f1(self) -> float:
        return 0.5 * (self.report_a.f1 + self.report_b.f1)

    @property
    def mean_leakage(self) -> float:
        return 0.5 * (self.report_a.shared_leakage + self.report_b.shared_leakage)

    def summary(self) -> str:
        return (
            f"{self.architecture:>8s} seed={self.seed} "
            f"recall={self.mean_recall:.3f} precision={self.mean_precision:.3f} "
            f"f1={self.mean_f1:.3f} leakage={self.mean_leakage:.3f} "
            f"fve={0.5 * (self.fve_a + self.fve_b):.3f}"
        )


def run_comparison(
    toy: ToyConfig,
    crosscoder: CrosscoderConfig,
    training: TrainConfig,
    architectures: tuple[str, ...] = ("dfc", "standard"),
    threshold: float = 0.5,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> list[ComparisonReport]:
    """Train each architecture on identical data and score against ground truth.

    Args:
        toy: Activation source; must contain genuinely exclusive concepts.
        crosscoder: Base configuration. ``exclusive_frac`` is overridden per
            architecture, so both see the same dictionary size and sparsity.
        training: Training configuration, identical across architectures.
        architectures: Which to run.
        threshold: Cosine similarity for a feature-concept match.
        seed: Seed applied to data, initialisation and training alike.
        progress: Optional callback for progress lines.

    Returns:
        One :class:`ComparisonReport` per architecture.

    Raises:
        ValueError: If the toy configuration has no exclusive concepts to find.
    """
    if toy.n_excl_a == 0 and toy.n_excl_b == 0 and not toy.drop_concepts:
        raise ValueError("comparison needs a regime with exclusive concepts")

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    results: list[ComparisonReport] = []
    for arch in architectures:
        frac = crosscoder.exclusive_frac if arch == "dfc" else 0.0
        cfg = CrosscoderConfig(**{**crosscoder.__dict__, "exclusive_frac": frac})

        # Rebuild the source per architecture so both see identical batches.
        source = ToyActivations(toy)
        torch.manual_seed(seed)
        model = Crosscoder(cfg)
        say(f"training {arch} (exclusive_frac={frac})")
        history = train(model, source.sample_activations, training)
        final = history[-1] if history else {"fve_a": 0.0, "fve_b": 0.0}

        results.append(
            ComparisonReport(
                architecture=arch,
                report_a=concept_recovery(model, source, "a", threshold),
                report_b=concept_recovery(model, source, "b", threshold),
                fve_a=final["fve_a"],
                fve_b=final["fve_b"],
                seed=seed,
            )
        )
        say("  " + results[-1].summary())

    return results
