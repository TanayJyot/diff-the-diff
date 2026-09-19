"""Training loop for crosscoders and DFCs.

Implements the preprocessing the paper reports as necessary rather than
optional: median-L2 activation normalisation so both models contribute
comparably to the joint loss, and per-batch outlier masking. The paper notes
Qwen produced activation norms up to ten times the median, concentrated at the
first token position; without masking, those tokens dominate the gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from diffdiff.diffing.crosscoder import Crosscoder

BatchFn = Callable[[int], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainConfig:
    """Attributes:
        steps: Optimizer steps.
        batch_size: Activations per step. BatchTopK holds pre-activations for
            the whole batch, so this trades directly against memory.
        lr: Adam learning rate.
        warmup_steps: Linear warmup.
        outlier_mult: Drop activations whose L2 norm exceeds this multiple of
            the batch median from the loss.
        log_every: Steps between metric records.
        device: Torch device.
        seed: RNG seed for the optimizer's initialisation.
    """

    steps: int = 2_000
    batch_size: int = 512
    lr: float = 1e-4
    warmup_steps: int = 100
    outlier_mult: float = 2.0
    log_every: int = 100
    device: str = "cpu"
    seed: int = 0


class ActivationNormalizer:
    """Scales each model's activations so their median L2 norm is
    ``sqrt((d_a + d_b) / 2)``.

    Without this, the model with larger-norm activations dominates the joint
    reconstruction loss and the dictionary allocates itself accordingly.
    """

    def __init__(self, x_a: torch.Tensor, x_b: torch.Tensor):
        target = math.sqrt((x_a.shape[-1] + x_b.shape[-1]) / 2)
        self.scale_a = target / x_a.norm(dim=-1).median().clamp_min(1e-8)
        self.scale_b = target / x_b.norm(dim=-1).median().clamp_min(1e-8)

    def __call__(
        self, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x_a * self.scale_a, x_b * self.scale_b


def mask_outliers(
    x_a: torch.Tensor, x_b: torch.Tensor, mult: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop rows whose norm exceeds ``mult`` times the batch median in either model."""
    if mult <= 0:
        return x_a, x_b
    na, nb = x_a.norm(dim=-1), x_b.norm(dim=-1)
    keep = (na <= mult * na.median()) & (nb <= mult * nb.median())
    if not keep.any():  # degenerate batch; better to train on it than to stall
        return x_a, x_b
    return x_a[keep], x_b[keep]


def train(
    model: Crosscoder,
    batch_fn: BatchFn,
    config: TrainConfig,
    progress: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Train a crosscoder in place.

    Args:
        model: The crosscoder or DFC to train.
        batch_fn: Called with a batch size, returns ``(x_a, x_b)``.
        config: Training configuration.
        progress: Optional callback invoked with ``(step, metrics)`` on each log.

    Returns:
        The recorded metric history.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)

    # Calibrate normalisation on a warmup sample rather than per batch, so the
    # scale is a fixed property of the data and not a moving target.
    cal_a, cal_b = batch_fn(min(4096, config.batch_size * 8))
    normalizer = ActivationNormalizer(cal_a.to(device), cal_b.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, config.warmup_steps))
    )

    history: list[dict[str, float]] = []
    for step in range(config.steps):
        x_a, x_b = batch_fn(config.batch_size)
        x_a, x_b = normalizer(x_a.to(device), x_b.to(device))
        x_a, x_b = mask_outliers(x_a, x_b, config.outlier_mult)

        loss, metrics = model.loss(x_a, x_b, step=step)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % config.log_every == 0 or step == config.steps - 1:
            metrics["step"] = step
            history.append(metrics)
            if progress is not None:
                progress(step, metrics)

    return history


@torch.no_grad()
def collect_features(
    model: Crosscoder,
    batch_fn: BatchFn,
    n_batches: int,
    batch_size: int,
    device: str = "cpu",
    normalizer: ActivationNormalizer | None = None,
) -> torch.Tensor:
    """Run the trained model over fresh data and return stacked feature activations.

    Used by the null and mirror analyses, which need firing patterns rather than
    weights.
    """
    model.eval()
    dev = torch.device(device)
    if normalizer is None:
        cal_a, cal_b = batch_fn(min(4096, batch_size * 8))
        normalizer = ActivationNormalizer(cal_a.to(dev), cal_b.to(dev))

    chunks = []
    for _ in range(n_batches):
        x_a, x_b = batch_fn(batch_size)
        x_a, x_b = normalizer(x_a.to(dev), x_b.to(dev))
        chunks.append(model.encode(x_a, x_b, model.config.k).cpu())
    model.train()
    return torch.cat(chunks, dim=0)
