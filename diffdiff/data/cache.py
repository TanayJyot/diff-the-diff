"""Disk-backed activation cache.

Activations do not fit in memory at any useful scale, and they do not fit in
VRAM at all on an 8GB card: 2M tokens of a 1024-wide residual stream in fp16 is
~4GB per model. Capture and training are therefore separate passes, with a
memory-mapped file between them, so that model weights are never resident while
the crosscoder trains.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CacheMeta:
    """Sidecar metadata describing a cache file.

    Attributes:
        model_name: Model the activations came from.
        layer: Layer index captured.
        d_model: Residual stream width.
        n_tokens: Number of activation rows.
        dtype: Numpy dtype name.
        transform: Which transform produced this model, if any.
    """

    model_name: str
    layer: int
    d_model: int
    n_tokens: int
    dtype: str = "float16"
    transform: str | None = None


class ActivationCache:
    """A memory-mapped ``(n_tokens, d_model)`` array of activations."""

    def __init__(self, path: str | Path, meta: CacheMeta, mode: str = "r"):
        self.path = Path(path)
        self.meta = meta
        self.array = np.memmap(
            self.path, dtype=meta.dtype, mode=mode, shape=(meta.n_tokens, meta.d_model)
        )

    @classmethod
    def create(cls, path: str | Path, meta: CacheMeta) -> "ActivationCache":
        """Allocate a new cache file and write its sidecar metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = cls(path, meta, mode="w+")
        cache._write_meta()
        return cache

    @classmethod
    def open(cls, path: str | Path) -> "ActivationCache":
        """Open an existing cache, reading its sidecar metadata."""
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"no metadata sidecar at {meta_path}")
        with open(meta_path) as fh:
            return cls(path, CacheMeta(**json.load(fh)), mode="r")

    def _write_meta(self) -> None:
        meta_path = self.path.with_suffix(self.path.suffix + ".json")
        with open(meta_path, "w") as fh:
            json.dump(asdict(self.meta), fh, indent=2)

    def write(self, start: int, activations: torch.Tensor) -> int:
        """Write a block of activations, returning the next write offset."""
        arr = activations.detach().to(torch.float32).cpu().numpy().astype(self.meta.dtype)
        end = min(start + arr.shape[0], self.meta.n_tokens)
        if end > start:
            self.array[start:end] = arr[: end - start]
        return end

    def flush(self) -> None:
        self.array.flush()

    def __len__(self) -> int:
        return self.meta.n_tokens


def paired_batch_fn(
    cache_a: ActivationCache,
    cache_b: ActivationCache,
    seed: int = 0,
    device: str = "cpu",
):
    """Build a ``batch_fn`` sampling aligned rows from two caches.

    Row ``i`` of each cache must refer to the same aligned position, which is
    what the aligner guarantees at capture time.

    Raises:
        ValueError: If the two caches differ in length.
    """
    if len(cache_a) != len(cache_b):
        raise ValueError(
            f"caches are not aligned: {len(cache_a)} vs {len(cache_b)} rows"
        )
    rng = np.random.default_rng(seed)
    n = len(cache_a)
    dev = torch.device(device)

    def batch_fn(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.sort(rng.choice(n, size=min(batch_size, n), replace=False))
        x_a = torch.from_numpy(np.asarray(cache_a.array[idx], dtype=np.float32))
        x_b = torch.from_numpy(np.asarray(cache_b.array[idx], dtype=np.float32))
        return x_a.to(dev), x_b.to(dev)

    return batch_fn
