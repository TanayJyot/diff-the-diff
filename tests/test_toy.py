import pytest
import torch

from diffdiff.data.toy import (
    ToyActivations,
    ToyConfig,
    cross_arch_config,
    null_config,
    quantized_config,
)


def test_null_config_gives_identical_activations():
    cfg = null_config(d=32, n_shared=64)
    assert cfg.is_null
    src = ToyActivations(cfg)
    x_a, x_b = src.sample_activations(128)
    assert torch.equal(x_a, x_b), "a null must be bit-identical, not merely close"


def test_quantized_config_is_near_but_not_identical():
    cfg = quantized_config(d=32, n_shared=64, noise_std=0.02)
    assert not cfg.is_null
    src = ToyActivations(cfg)
    x_a, x_b = src.sample_activations(256)
    assert not torch.equal(x_a, x_b)
    rel = (x_a - x_b).norm() / x_a.norm()
    assert rel < 0.2, "quantization noise should be small relative to signal"


def test_dropped_concept_is_invisible_to_b_only():
    cfg = quantized_config(d=32, n_shared=64, drop_concepts=(5,))
    src = ToyActivations(cfg)
    gt = src.ground_truth()
    assert gt["visible_a"][5]
    assert not gt["visible_b"][5]


def test_drop_index_must_be_shared():
    with pytest.raises(ValueError):
        ToyActivations(ToyConfig(d_a=16, d_b=16, n_shared=8, drop_concepts=(99,)))


def test_cross_arch_shapes_and_exclusives():
    cfg = cross_arch_config(d_a=32, d_b=24, n_shared=48, n_excl=8)
    src = ToyActivations(cfg)
    x_a, x_b = src.sample_activations(16)
    assert x_a.shape == (16, 32)
    assert x_b.shape == (16, 24)
    gt = src.ground_truth()
    assert len(gt["excl_a"]) == 8
    assert len(gt["excl_b"]) == 8
    # Exclusive concepts are visible to exactly one model.
    assert gt["visible_a"][gt["excl_a"]].all()
    assert not gt["visible_b"][gt["excl_a"]].any()


def test_identity_transform_requires_matching_dims():
    with pytest.raises(ValueError):
        ToyActivations(ToyConfig(d_a=16, d_b=8, transform="identity"))


def test_unknown_transform_rejected():
    with pytest.raises(ValueError):
        ToyActivations(ToyConfig(d_a=16, d_b=16, transform="nonsense"))


def test_seed_is_reproducible():
    a = ToyActivations(null_config(d=16, n_shared=32, seed=7)).sample_activations(32)[0]
    b = ToyActivations(null_config(d=16, n_shared=32, seed=7)).sample_activations(32)[0]
    assert torch.equal(a, b)


def test_activations_are_sparse_combinations():
    cfg = null_config(d=64, n_shared=128)
    src = ToyActivations(cfg)
    x_a, _ = src.sample_activations(256)
    assert x_a.norm(dim=-1).median() > 0, "activations should not be degenerate"
