"""The model transformations under test.

Each transform takes a model reference and returns a second
:class:`~diffdiff.models.loader.ModelHandle` to diff against the first. The
identity transform is what the null test uses on real models: it must produce a
diff of nothing.
"""

from __future__ import annotations

from typing import Any

from diffdiff.models.loader import ModelHandle, load_model

#: Transformations ordered by how destructive they are. The monotonicity test
#: (``docs/DESIGN.md`` §6.2) asserts that measured diff magnitude follows this
#: order; a method that inverts it is measuring noise.
DAMAGE_ORDER = ("identity", "int8", "nf4")


def apply_transform(
    name: str,
    model_name: str,
    layer: int | str = "middle",
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> ModelHandle:
    """Load the transformed counterpart of a model.

    Args:
        name: One of :data:`DAMAGE_ORDER`.
        model_name: Hub id or path of the source model.
        layer: Layer to capture.
        device: Device map target.
        dtype: Compute dtype.

    Returns:
        A handle to the transformed model.
    """
    if name == "identity":
        return load_model(model_name, layer=layer, dtype=dtype, device=device)
    if name in ("int8", "nf4"):
        return load_model(
            model_name, layer=layer, dtype=dtype, device=device, quantization=name
        )
    raise ValueError(f"unknown transform {name!r}; expected one of {DAMAGE_ORDER}")


def merge_state_dicts(
    a: dict[str, Any], b: dict[str, Any], weight: float = 0.5
) -> dict[str, Any]:
    """Linearly interpolate two state dicts (a "model soup" merge).

    Args:
        a: First state dict.
        b: Second, with identical keys and shapes.
        weight: Weight on ``b``.

    Returns:
        The merged state dict.

    Raises:
        ValueError: If the two state dicts do not match.
    """
    if a.keys() != b.keys():
        missing = set(a) ^ set(b)
        raise ValueError(f"state dicts differ in {len(missing)} keys, e.g. {sorted(missing)[:3]}")
    merged = {}
    for key, va in a.items():
        vb = b[key]
        if va.shape != vb.shape:
            raise ValueError(f"shape mismatch at {key}: {va.shape} vs {vb.shape}")
        merged[key] = (1.0 - weight) * va + weight * vb
    return merged
