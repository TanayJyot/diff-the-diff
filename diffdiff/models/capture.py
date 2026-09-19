"""Capturing residual stream activations into a cache.

One forward pass per model, hooked at the chosen layer, writing aligned rows to
a memory-mapped file. Running the two models in separate passes rather than
together is what keeps peak memory at one model rather than two.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from diffdiff.align.aligner import Aligner, IdentityAligner
from diffdiff.data.cache import ActivationCache, CacheMeta
from diffdiff.models.loader import ModelHandle, _transformer_blocks


@torch.no_grad()
def capture_pair(
    handle_a: ModelHandle,
    handle_b: ModelHandle,
    texts: Iterable[str],
    cache_path_a: str,
    cache_path_b: str,
    n_tokens: int,
    aligner: Aligner | None = None,
    max_length: int = 512,
    transform: str | None = None,
) -> tuple[ActivationCache, ActivationCache]:
    """Capture aligned activations from two models over the same texts.

    Args:
        handle_a: First model.
        handle_b: Second model.
        texts: Corpus to run through both models.
        cache_path_a: Destination file for model A.
        cache_path_b: Destination file for model B.
        n_tokens: Number of aligned activation rows to collect.
        aligner: Token aligner. Defaults to :class:`IdentityAligner`, correct
            whenever the two models share a tokenizer.
        max_length: Truncation length per text.
        transform: Recorded in model B's metadata, for reporting.

    Returns:
        The two caches, both containing exactly ``n_tokens`` aligned rows.
    """
    aligner = aligner or IdentityAligner()
    cache_a = ActivationCache.create(
        cache_path_a,
        CacheMeta(handle_a.name, handle_a.layer, handle_a.d_model, n_tokens),
    )
    cache_b = ActivationCache.create(
        cache_path_b,
        CacheMeta(handle_b.name, handle_b.layer, handle_b.d_model, n_tokens, transform=transform),
    )

    written = 0
    for text in texts:
        if written >= n_tokens:
            break
        acts_a, tokens_a = _forward_capture(handle_a, text, max_length)
        acts_b, tokens_b = _forward_capture(handle_b, text, max_length)

        pairs = aligner.align(tokens_a, tokens_b)
        if not pairs:
            continue
        idx_a = torch.tensor([p[0] for p in pairs])
        idx_b = torch.tensor([p[1] for p in pairs])

        written_next = cache_a.write(written, acts_a[idx_a])
        cache_b.write(written, acts_b[idx_b])
        written = written_next

    cache_a.flush()
    cache_b.flush()
    if written < n_tokens:
        raise ValueError(
            f"corpus exhausted after {written} of {n_tokens} requested tokens; "
            "supply more text"
        )
    return cache_a, cache_b


@torch.no_grad()
def _forward_capture(
    handle: ModelHandle, text: str, max_length: int
) -> tuple[torch.Tensor, Sequence[int]]:
    """Run one text and return ``(activations, token_ids)`` for the hooked layer."""
    enc = handle.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    device = next(handle.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # Transformer blocks return either a tensor or a tuple whose first
        # element is the hidden state.
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach()[0].float().cpu())

    block = _transformer_blocks(handle.model)[handle.layer]
    fired = block.register_forward_hook(hook)
    try:
        handle.model(**enc)
    finally:
        fired.remove()

    if not captured:
        raise RuntimeError(f"hook on layer {handle.layer} captured nothing")
    return captured[0], enc["input_ids"][0].tolist()
