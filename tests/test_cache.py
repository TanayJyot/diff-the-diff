import numpy as np
import pytest
import torch

from diffdiff.data.cache import ActivationCache, CacheMeta, paired_batch_fn


def make_cache(tmp_path, name, n_tokens=64, d_model=8):
    meta = CacheMeta(model_name=name, layer=3, d_model=d_model, n_tokens=n_tokens)
    cache = ActivationCache.create(tmp_path / f"{name}.bin", meta)
    cache.write(0, torch.arange(n_tokens * d_model, dtype=torch.float32).reshape(n_tokens, d_model))
    cache.flush()
    return cache


def test_cache_roundtrip(tmp_path):
    make_cache(tmp_path, "a")
    reopened = ActivationCache.open(tmp_path / "a.bin")
    assert len(reopened) == 64
    assert reopened.meta.layer == 3
    assert reopened.meta.model_name == "a"
    assert np.allclose(reopened.array[0], np.arange(8))


def test_cache_write_respects_capacity(tmp_path):
    meta = CacheMeta("m", 0, 4, n_tokens=10)
    cache = ActivationCache.create(tmp_path / "m.bin", meta)
    end = cache.write(8, torch.ones(5, 4))
    assert end == 10, "writes must clamp to the declared capacity"


def test_open_missing_metadata_raises(tmp_path):
    (tmp_path / "orphan.bin").write_bytes(b"\x00" * 16)
    with pytest.raises(FileNotFoundError, match="metadata sidecar"):
        ActivationCache.open(tmp_path / "orphan.bin")


def test_paired_batch_fn_shapes_and_alignment(tmp_path):
    a = make_cache(tmp_path, "a")
    b = make_cache(tmp_path, "b")
    batch_fn = paired_batch_fn(a, b, seed=0)
    x_a, x_b = batch_fn(16)
    assert x_a.shape == (16, 8)
    assert torch.equal(x_a, x_b), "identical caches must yield identical rows"


def test_paired_batch_fn_rejects_mismatched_lengths(tmp_path):
    a = make_cache(tmp_path, "a", n_tokens=64)
    b = make_cache(tmp_path, "b", n_tokens=32)
    with pytest.raises(ValueError, match="not aligned"):
        paired_batch_fn(a, b)


def test_paired_batch_fn_caps_at_cache_size(tmp_path):
    a = make_cache(tmp_path, "a", n_tokens=10)
    b = make_cache(tmp_path, "b", n_tokens=10)
    x_a, _ = paired_batch_fn(a, b)(999)
    assert x_a.shape[0] == 10
