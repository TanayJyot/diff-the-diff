"""Loading models and resolving which layer to diff.

Kept deliberately thin: everything here is a wrapper over ``transformers`` so
that the diffing code never imports it directly, and the toy path stays usable
without the dependency installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelHandle:
    """A loaded model plus the metadata the harness needs.

    Attributes:
        model: The loaded ``PreTrainedModel``.
        tokenizer: Its tokenizer.
        name: Hub id or local path, for reporting.
        layer: Index of the layer whose residual stream we capture.
        d_model: Residual stream width.
    """

    model: Any
    tokenizer: Any
    name: str
    layer: int
    d_model: int


def resolve_layer(n_layers: int, layer: int | str = "middle") -> int:
    """Resolve a layer specification to an index.

    ``"middle"`` follows the paper, which captures the middle-layer residual
    stream. Note the paper reports no sensitivity analysis over this choice, so
    it is a convention rather than a validated default — see ``docs/DESIGN.md``
    §10.
    """
    if layer == "middle":
        return n_layers // 2
    if isinstance(layer, str):
        raise ValueError(f"unknown layer spec: {layer!r}")
    if not -n_layers <= layer < n_layers:
        raise ValueError(f"layer {layer} out of range for {n_layers} layers")
    return layer % n_layers


def load_model(
    name: str,
    layer: int | str = "middle",
    dtype: str = "bfloat16",
    device: str = "cuda",
    quantization: str | None = None,
) -> ModelHandle:
    """Load a causal LM, optionally quantized.

    Args:
        name: Hub id or local path.
        layer: Layer index or ``"middle"``.
        dtype: Torch dtype name for the unquantized case.
        device: Device map target.
        quantization: ``None``, ``"int8"`` or ``"nf4"``. The latter two require
            ``bitsandbytes``.

    Returns:
        A :class:`ModelHandle`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "device_map": device}
    if quantization is not None:
        from transformers import BitsAndBytesConfig

        if quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype),
            )
        else:
            raise ValueError(f"unknown quantization: {quantization!r}")
        kwargs.pop("dtype", None)

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)

    blocks = _transformer_blocks(model)
    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        name=name,
        layer=resolve_layer(len(blocks), layer),
        d_model=model.config.hidden_size,
    )


def _transformer_blocks(model: Any) -> Any:
    """Find the list of transformer blocks across common architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        "add its attribute path to _transformer_blocks"
    )
