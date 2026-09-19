"""BatchTopK crosscoders, with optional dedicated (DFC) feature partitions.

A crosscoder learns one sparse dictionary bridging the activation spaces of two
models ``A`` and ``B``, which may have different hidden dimensions. Both models
encode into a shared latent space; the pre-activations are averaged, sparsified
with BatchTopK, and decoded back into each model's own space.

Two architectures share this implementation, distinguished only by how the
dictionary is partitioned:

**Standard crosscoder** — every feature reconstructs both models. Exclusivity is
read off *post hoc* via the relative decoder norm
``R_i = ||d_i^A|| / (||d_i^A|| + ||d_i^B||)``, where ``R_i ~ 1`` means
A-exclusive. Because the joint objective rewards features that serve both
models, this architecture carries a prior *against* discovering exclusive
features.

**Dedicated Feature Crosscoder (DFC)** — the dictionary is partitioned into
three disjoint sets ``I_A``, ``I_S``, ``I_B``, and the opposing decoder weights
are structurally absent rather than merely small. A feature in ``I_A`` therefore
satisfies ``||d_i^B||_2 == 0`` *exactly*, and receives no gradient from model
B's reconstruction error.

Feature indices are laid out as ``[I_A | I_S | I_B]`` so that both decoders read
a contiguous slice: model A decodes features ``[0, n_A + n_S)`` and model B
decodes features ``[n_A, n_features)``.

Reference: Jiralerspong & Bricken, "Cross-Architecture Model Diffing with
Crosscoders" (arXiv:2602.11729).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_topk(acts: torch.Tensor, k: int) -> torch.Tensor:
    """Zero all but the ``k * batch_size`` largest activations in the batch.

    BatchTopK enforces a true L0 constraint on average rather than per sample,
    so a token that genuinely needs many features may use them, paying for it
    with tokens that need fewer. This avoids the shrinkage and decoupling
    artifacts that an L1 penalty introduces (arXiv:2504.02922).

    Args:
        acts: Non-negative activations, shape ``(batch, n_features)``.
        k: Average number of active features per sample.

    Returns:
        Tensor of the same shape, with all but the top ``k * batch`` entries
        zeroed.
    """
    if k <= 0:
        return torch.zeros_like(acts)

    batch, n_features = acts.shape
    n_keep = min(k * batch, acts.numel())

    flat = acts.flatten()
    # `topk` on the flattened batch is what makes this *batch* top-k: the budget
    # is shared across samples rather than allocated per sample.
    threshold = torch.topk(flat, n_keep, sorted=True).values[-1]
    return torch.where(acts >= threshold, acts, torch.zeros_like(acts))


@dataclass
class CrosscoderConfig:
    """Configuration for a crosscoder or DFC.

    Defaults follow the hyperparameters reported in arXiv:2602.11729 §B.1 where
    they are scale-independent. Sizes are left to the caller because the paper's
    dictionary (131,072) assumes 8-20B models on an H100.

    Attributes:
        d_a: Hidden dimension of model A.
        d_b: Hidden dimension of model B.
        n_features: Total dictionary size, across all three partitions.
        exclusive_frac: Fraction of the dictionary dedicated to *each* model's
            exclusive partition. ``0.0`` gives a standard crosscoder (no
            dedicated partitions); ``0.05`` reproduces the paper's main DFC.
            Note the paper found this materially affects which features are
            discovered: 1% and 3% partitions found broad features but missed
            granular ones.
        k: Final BatchTopK sparsity.
        k_initial: Sparsity at the start of annealing. Features form more
            effectively before the sparsity objective is fully enforced.
        anneal_steps: Steps over which ``k`` decays from ``k_initial`` to ``k``.
        aux_alpha: Weight of the AuxK dead-feature loss.
        aux_k: Number of dead features used by the AuxK loss.
        dead_after_tokens: A feature is considered dead once it has not fired
            for this many consecutive tokens.
        dec_init_norm: Initial L2 norm of each decoder vector.
    """

    d_a: int
    d_b: int
    n_features: int = 16384
    exclusive_frac: float = 0.05
    k: int = 32
    k_initial: int = 128
    anneal_steps: int = 5_000
    aux_alpha: float = 0.03
    aux_k: int = 256
    dead_after_tokens: int = 1_000_000
    dec_init_norm: float = 0.4

    @property
    def is_dfc(self) -> bool:
        return self.exclusive_frac > 0.0

    @property
    def n_exclusive(self) -> int:
        """Size of each exclusive partition (``I_A`` and ``I_B`` are equal)."""
        return int(self.n_features * self.exclusive_frac)

    @property
    def n_shared(self) -> int:
        return self.n_features - 2 * self.n_exclusive

    def __post_init__(self) -> None:
        if not 0.0 <= self.exclusive_frac < 0.5:
            raise ValueError(
                f"exclusive_frac must be in [0, 0.5), got {self.exclusive_frac}"
            )
        if self.n_shared <= 0:
            raise ValueError(
                f"exclusive_frac={self.exclusive_frac} leaves no shared features"
            )
        if self.k > self.n_features:
            raise ValueError(f"k={self.k} exceeds n_features={self.n_features}")


class Crosscoder(nn.Module):
    """A BatchTopK crosscoder over two activation spaces.

    Set ``config.exclusive_frac > 0`` for a DFC; leave it at ``0`` for a
    standard crosscoder. The two differ only in whether the decoders span the
    full dictionary or a partition of it, so every downstream metric applies
    unchanged to both.
    """

    def __init__(self, config: CrosscoderConfig):
        super().__init__()
        self.config = config

        n_a, n_s = config.n_exclusive, config.n_shared
        # Layout [I_A | I_S | I_B] keeps both decoder slices contiguous.
        self.slice_a = slice(0, n_a + n_s)
        self.slice_b = slice(n_a, config.n_features)
        self.idx_excl_a = (0, n_a)
        self.idx_shared = (n_a, n_a + n_s)
        self.idx_excl_b = (n_a + n_s, config.n_features)

        # Encoders read the full dictionary from each model. Only the *decoder*
        # is severed in a DFC, matching the paper's definition of exclusivity as
        # a statement about reconstruction.
        self.W_enc_a = nn.Parameter(torch.empty(config.d_a, config.n_features))
        self.W_enc_b = nn.Parameter(torch.empty(config.d_b, config.n_features))
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))

        self.W_dec_a = nn.Parameter(torch.empty(self.slice_a.stop - self.slice_a.start, config.d_a))
        self.W_dec_b = nn.Parameter(torch.empty(self.slice_b.stop - self.slice_b.start, config.d_b))
        self.b_dec_a = nn.Parameter(torch.zeros(config.d_a))
        self.b_dec_b = nn.Parameter(torch.zeros(config.d_b))

        # Tokens since each feature last fired, for dead-feature detection.
        self.register_buffer("tokens_since_fired", torch.zeros(config.n_features))

        self._init_weights()

    def _init_weights(self) -> None:
        cfg = self.config
        for W in (self.W_dec_a, self.W_dec_b):
            nn.init.normal_(W)
            with torch.no_grad():
                W *= cfg.dec_init_norm / W.norm(dim=-1, keepdim=True)
        # Initialise encoders as the transpose of their decoders, restricted to
        # the features that model actually reconstructs. Features outside a
        # model's partition get a small random encoder instead.
        with torch.no_grad():
            nn.init.normal_(self.W_enc_a, std=0.02)
            nn.init.normal_(self.W_enc_b, std=0.02)
            self.W_enc_a[:, self.slice_a] = self.W_dec_a.T.clone()
            self.W_enc_b[:, self.slice_b] = self.W_dec_b.T.clone()

    def current_k(self, step: int) -> int:
        """Sparsity at ``step``, linearly annealed from ``k_initial`` to ``k``."""
        cfg = self.config
        if step >= cfg.anneal_steps or cfg.anneal_steps <= 0:
            return cfg.k
        frac = step / cfg.anneal_steps
        return int(round(cfg.k_initial + frac * (cfg.k - cfg.k_initial)))

    def encode(self, x_a: torch.Tensor, x_b: torch.Tensor, k: int) -> torch.Tensor:
        """Encode both models' activations into shared sparse features."""
        pre = 0.5 * (x_a @ self.W_enc_a + x_b @ self.W_enc_b) + self.b_enc
        return batch_topk(F.relu(pre), k)

    def decode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct each model from the features its partition spans."""
        recon_a = features[:, self.slice_a] @ self.W_dec_a + self.b_dec_a
        recon_b = features[:, self.slice_b] @ self.W_dec_b + self.b_dec_b
        return recon_a, recon_b

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor, step: int = 10**9
    ) -> dict[str, torch.Tensor]:
        k = self.current_k(step)
        features = self.encode(x_a, x_b, k)
        recon_a, recon_b = self.decode(features)
        return {"features": features, "recon_a": recon_a, "recon_b": recon_b}

    def decoder_norms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-feature decoder L2 norms in each model's space.

        Features outside a model's partition report exactly zero, which is the
        DFC's structural guarantee rather than an empirical observation.
        """
        cfg = self.config
        norms_a = torch.zeros(cfg.n_features, device=self.W_dec_a.device)
        norms_b = torch.zeros(cfg.n_features, device=self.W_dec_b.device)
        norms_a[self.slice_a] = self.W_dec_a.detach().norm(dim=-1)
        norms_b[self.slice_b] = self.W_dec_b.detach().norm(dim=-1)
        return norms_a, norms_b

    def full_decoder(self, model: str) -> torch.Tensor:
        """Decoder matrix over the *whole* dictionary, zero-padded.

        A DFC's decoders span only their own partition, so their row indices do
        not line up with feature indices. This returns a
        ``(n_features, d_model)`` matrix indexed by feature, with exact zeros
        for features that do not reconstruct this model — which lets callers
        index by feature id without tracking partition offsets.

        Args:
            model: ``"a"`` or ``"b"``.
        """
        cfg = self.config
        if model == "a":
            W, sl, d = self.W_dec_a, self.slice_a, cfg.d_a
        elif model == "b":
            W, sl, d = self.W_dec_b, self.slice_b, cfg.d_b
        else:
            raise ValueError(f"model must be 'a' or 'b', got {model!r}")
        full = torch.zeros(cfg.n_features, d, device=W.device, dtype=W.dtype)
        full[sl] = W.detach()
        return full

    def exclusive_mask(self, model: str) -> torch.Tensor:
        """Which features this architecture labels exclusive to ``model``.

        For a DFC this is the dedicated partition. For a standard crosscoder
        there is no partition, so we take the features with the most extreme
        relative decoder norm, budgeted to the *same count* a DFC would
        allocate — the comparison the paper makes when it takes the 500 most
        extreme features. Without matching the budget, recall and
        false-positive rates would not be comparable across architectures.
        """
        cfg = self.config
        if cfg.is_dfc:
            return self.partition_mask("A" if model == "a" else "B")

        budget = max(1, int(cfg.n_features * 0.05))
        rel = self.relative_decoder_norm()
        # R ~ 1 means A-exclusive, R ~ 0 means B-exclusive.
        scores = rel if model == "a" else -rel
        idx = torch.topk(scores, budget).indices
        mask = torch.zeros(cfg.n_features, dtype=torch.bool, device=rel.device)
        mask[idx] = True
        return mask

    def relative_decoder_norm(self, eps: float = 1e-8) -> torch.Tensor:
        """``R_i = ||d_i^A|| / (||d_i^A|| + ||d_i^B||)``, the standard crosscoder's
        post-hoc exclusivity measure. ``R_i ~ 1`` means A-exclusive, ``~0`` means
        B-exclusive, ``~0.5`` means shared.

        Decoder norms are not directly comparable across models with different
        hidden dimensions, so we normalise each model's norms by their median
        before taking the ratio. For ``d_a == d_b`` this is a monotone rescaling
        and changes nothing.
        """
        norms_a, norms_b = self.decoder_norms()
        scale_a = norms_a[self.slice_a].median().clamp_min(eps)
        scale_b = norms_b[self.slice_b].median().clamp_min(eps)
        na, nb = norms_a / scale_a, norms_b / scale_b
        return na / (na + nb + eps)

    def partition_of(self, index: int) -> str:
        """Return ``"A"``, ``"S"`` or ``"B"`` for a feature index."""
        if index < self.idx_excl_a[1]:
            return "A"
        if index < self.idx_shared[1]:
            return "S"
        return "B"

    def partition_mask(self, name: str) -> torch.Tensor:
        """Boolean mask selecting one partition of the dictionary."""
        lo, hi = {"A": self.idx_excl_a, "S": self.idx_shared, "B": self.idx_excl_b}[name]
        mask = torch.zeros(self.config.n_features, dtype=torch.bool, device=self.W_dec_a.device)
        mask[lo:hi] = True
        return mask

    @torch.no_grad()
    def update_dead_features(self, features: torch.Tensor) -> None:
        """Track tokens elapsed since each feature last fired."""
        fired = (features > 0).any(dim=0)
        self.tokens_since_fired += features.shape[0]
        self.tokens_since_fired[fired] = 0.0

    def dead_mask(self) -> torch.Tensor:
        return self.tokens_since_fired >= self.config.dead_after_tokens

    def aux_loss(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        recon_a: torch.Tensor,
        recon_b: torch.Tensor,
    ) -> torch.Tensor:
        """AuxK loss: ask dead features to explain the current residual.

        Following Gao et al. (2024). Without this a large fraction of the
        dictionary silently stops firing, which wastes capacity — and in a DFC
        would quietly shrink the exclusive partitions we are trying to measure.
        """
        dead = self.dead_mask()
        n_dead = int(dead.sum())
        if n_dead == 0:
            return x_a.new_zeros(())

        pre = 0.5 * (x_a @ self.W_enc_a + x_b @ self.W_enc_b) + self.b_enc
        pre = F.relu(pre)
        masked = torch.where(dead, pre, torch.zeros_like(pre))
        aux_feats = batch_topk(masked, min(self.config.aux_k, n_dead))

        err_a = (x_a - recon_a).detach()
        err_b = (x_b - recon_b).detach()
        aux_a, aux_b = self.decode(aux_feats)
        # Subtract the decoder biases: they are already in `recon`, so including
        # them here would ask the dead features to re-explain a constant.
        return (
            F.mse_loss(aux_a - self.b_dec_a, err_a)
            + F.mse_loss(aux_b - self.b_dec_b, err_b)
        )

    def loss(
        self, x_a: torch.Tensor, x_b: torch.Tensor, step: int = 10**9
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Joint reconstruction loss plus the AuxK term."""
        out = self.forward(x_a, x_b, step=step)
        recon_a, recon_b = out["recon_a"], out["recon_b"]

        mse_a = F.mse_loss(recon_a, x_a)
        mse_b = F.mse_loss(recon_b, x_b)
        aux = self.aux_loss(x_a, x_b, recon_a, recon_b)
        total = mse_a + mse_b + self.config.aux_alpha * aux

        self.update_dead_features(out["features"])

        with torch.no_grad():
            l0 = (out["features"] > 0).float().sum(dim=-1).mean()
            metrics = {
                "loss": float(total),
                "mse_a": float(mse_a),
                "mse_b": float(mse_b),
                "aux": float(aux),
                "l0": float(l0),
                "k": float(self.current_k(step)),
                "dead_frac": float(self.dead_mask().float().mean()),
                "fve_a": float(fraction_variance_explained(x_a, recon_a)),
                "fve_b": float(fraction_variance_explained(x_b, recon_b)),
            }
        return total, metrics


def fraction_variance_explained(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """R^2 of the reconstruction, the paper's headline quality metric."""
    resid = (x - recon).pow(2).sum()
    total = (x - x.mean(dim=0, keepdim=True)).pow(2).sum()
    return 1.0 - resid / total.clamp_min(1e-8)
