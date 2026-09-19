"""Synthetic activations with ground-truth concepts.

This reproduces the toy model of arXiv:2602.11729 §2.3 and extends it with the
two regimes our use case actually needs.

Ground-truth concepts are random unit vectors, partitioned into shared,
A-exclusive and B-exclusive sets. Each sample activates a sparse, correlated
subset. Model A's activation is the linear combination of the concepts A can
see; model B's is built the same way from B's concepts, optionally passed
through a random affine map to simulate the representational misalignment
between two different architectures.

Three regimes matter here:

``null``
    No exclusive concepts, identity transform, no noise, so ``x_a`` and ``x_b``
    are *bit-identical*. Any feature a diffing method places in an exclusive
    partition is, by construction, a false positive. This is the experiment the
    paper does not run.

``quantized``
    Identity transform plus small isotropic noise, optionally dropping a named
    concept outright. This is the near-identical regime our quantization use
    case lives in: the two models differ by rounding error, and possibly by one
    genuinely destroyed capability that we know the identity of.

``cross_arch``
    Genuine exclusive concepts on both sides and a random affine transform on
    B's concept vectors. This is the paper's own setting, and our control that
    the harness can find differences when differences exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ToyConfig:
    """Configuration for the synthetic concept model.

    Attributes:
        d_a: Activation dimension of model A.
        d_b: Activation dimension of model B.
        n_shared: Concepts visible to both models.
        n_excl_a: Concepts visible only to model A.
        n_excl_b: Concepts visible only to model B.
        n_active: Expected number of concepts active per sample.
        freq_alpha: Power-law exponent for concept frequency. Larger values make
            a few concepts dominate, as in real activations.
        n_clusters: Concepts are grouped into co-activating clusters; drawing one
            member raises the chance of drawing its siblings.
        cluster_p: Probability a sibling co-activates.
        transform: ``"identity"`` (same representational space) or ``"affine"``
            (random rotation + scaling, simulating a different architecture).
        noise_std: Isotropic Gaussian noise added to ``x_b`` only, as a fraction
            of the activation's typical scale. Models quantization error.
        drop_concepts: Indices of shared concepts removed from ``x_b`` only.
            These are known-destroyed capabilities: the answer key for recall.
        seed: RNG seed.
    """

    d_a: int = 128
    d_b: int = 128
    n_shared: int = 256
    n_excl_a: int = 0
    n_excl_b: int = 0
    n_active: int = 8
    freq_alpha: float = 1.0
    n_clusters: int = 16
    cluster_p: float = 0.3
    transform: str = "identity"
    noise_std: float = 0.0
    drop_concepts: tuple[int, ...] = field(default_factory=tuple)
    seed: int = 0

    @property
    def n_concepts(self) -> int:
        return self.n_shared + self.n_excl_a + self.n_excl_b

    @property
    def is_null(self) -> bool:
        """True when ``x_a`` and ``x_b`` are identical by construction."""
        return (
            self.n_excl_a == 0
            and self.n_excl_b == 0
            and self.transform == "identity"
            and self.noise_std == 0.0
            and not self.drop_concepts
            and self.d_a == self.d_b
        )


class ToyActivations:
    """Generates paired activations with known ground-truth concept structure.

    Concept indices are laid out as ``[shared | excl_a | excl_b]``.
    """

    def __init__(self, config: ToyConfig, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        self._gen = gen

        n = config.n_concepts
        # Ground-truth concepts: random unit vectors, isotropically distributed
        # so no direction is privileged.
        concepts = torch.randn(n, config.d_a, generator=gen)
        self.concepts_a = concepts / concepts.norm(dim=-1, keepdim=True)

        if config.transform == "identity":
            if config.d_a != config.d_b:
                raise ValueError("identity transform requires d_a == d_b")
            self.concepts_b = self.concepts_a.clone()
        elif config.transform == "affine":
            # A random affine map stands in for the fact that two architectures
            # encode the same concept along unrelated directions.
            M = torch.randn(config.d_a, config.d_b, generator=gen) / (config.d_a ** 0.5)
            mapped = self.concepts_a @ M
            self.concepts_b = mapped / mapped.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown transform: {config.transform!r}")

        # Visibility: shared concepts to both, exclusives to their own model.
        self.visible_a = torch.zeros(n, dtype=torch.bool)
        self.visible_b = torch.zeros(n, dtype=torch.bool)
        s, ea = config.n_shared, config.n_excl_a
        self.visible_a[:s] = True
        self.visible_b[:s] = True
        self.visible_a[s : s + ea] = True
        self.visible_b[s + ea :] = True

        # A dropped concept is visible to A but destroyed in B. This is the
        # implanted-capability answer key.
        for idx in config.drop_concepts:
            if not 0 <= idx < s:
                raise ValueError(f"drop_concepts index {idx} is not a shared concept")
            self.visible_b[idx] = False

        # Power-law frequencies, normalised so `n_active` concepts fire on average.
        ranks = torch.arange(1, n + 1, dtype=torch.float32)
        freqs = ranks.pow(-config.freq_alpha)
        self.freqs = (freqs / freqs.sum() * config.n_active).clamp(max=0.95)

        # Co-activation clusters give the data realistic correlational structure.
        self.cluster_of = torch.randint(0, config.n_clusters, (n,), generator=gen)

        self.concepts_a = self.concepts_a.to(self.device)
        self.concepts_b = self.concepts_b.to(self.device)

    def sample_activations(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a batch of paired activations.

        Returns:
            ``(x_a, x_b)`` of shapes ``(batch, d_a)`` and ``(batch, d_b)``.
        """
        cfg = self.config
        n = cfg.n_concepts
        gen = self._gen

        active = torch.rand(batch, n, generator=gen) < self.freqs

        # Co-activation: for each cluster, if anything in it fired, give its
        # other members an extra chance to fire too.
        cluster_hit = torch.zeros(batch, cfg.n_clusters, dtype=torch.bool)
        for c in range(cfg.n_clusters):
            members = self.cluster_of == c
            if members.any():
                cluster_hit[:, c] = active[:, members].any(dim=-1)
        sibling = cluster_hit[:, self.cluster_of]
        extra = (torch.rand(batch, n, generator=gen) < cfg.cluster_p) & sibling
        active = active | extra

        # Positive magnitudes, as activations of real features are non-negative.
        coeffs = active.float() * torch.rand(batch, n, generator=gen).add(0.5)
        coeffs = coeffs.to(self.device)

        x_a = (coeffs * self.visible_a.to(self.device).float()) @ self.concepts_a
        x_b = (coeffs * self.visible_b.to(self.device).float()) @ self.concepts_b

        if cfg.noise_std > 0:
            scale = x_b.norm(dim=-1, keepdim=True).median()
            noise = torch.randn(x_b.shape, generator=gen).to(self.device)
            x_b = x_b + noise * cfg.noise_std * scale / (cfg.d_b ** 0.5)

        return x_a, x_b

    def ground_truth(self) -> dict[str, torch.Tensor]:
        """Concept-level answer key, for scoring recall and false positives."""
        cfg = self.config
        s, ea = cfg.n_shared, cfg.n_excl_a
        return {
            "shared": torch.arange(0, s),
            "excl_a": torch.arange(s, s + ea),
            "excl_b": torch.arange(s + ea, cfg.n_concepts),
            "dropped": torch.tensor(cfg.drop_concepts, dtype=torch.long),
            "visible_a": self.visible_a,
            "visible_b": self.visible_b,
        }


def null_config(d: int = 128, n_shared: int = 256, seed: int = 0) -> ToyConfig:
    """A perfect null: ``x_a`` and ``x_b`` are bit-identical."""
    return ToyConfig(
        d_a=d, d_b=d, n_shared=n_shared, n_excl_a=0, n_excl_b=0,
        transform="identity", noise_std=0.0, seed=seed,
    )


def quantized_config(
    d: int = 128,
    n_shared: int = 256,
    noise_std: float = 0.02,
    drop_concepts: tuple[int, ...] = (),
    seed: int = 0,
) -> ToyConfig:
    """Near-identical models differing by rounding error, and optionally by a
    known set of destroyed concepts."""
    return ToyConfig(
        d_a=d, d_b=d, n_shared=n_shared, n_excl_a=0, n_excl_b=0,
        transform="identity", noise_std=noise_std,
        drop_concepts=drop_concepts, seed=seed,
    )


def cross_arch_config(
    d_a: int = 128, d_b: int = 96, n_shared: int = 224,
    n_excl: int = 16, seed: int = 0,
) -> ToyConfig:
    """Two different architectures with genuinely exclusive concepts — the
    positive control that the harness finds differences when they exist."""
    return ToyConfig(
        d_a=d_a, d_b=d_b, n_shared=n_shared, n_excl_a=n_excl, n_excl_b=n_excl,
        transform="affine", noise_std=0.0, seed=seed,
    )
