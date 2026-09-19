import pytest
import torch

from diffdiff.diffing.crosscoder import (
    Crosscoder,
    CrosscoderConfig,
    batch_topk,
    fraction_variance_explained,
)


def test_batch_topk_keeps_k_times_batch_entries():
    acts = torch.rand(8, 64)
    out = batch_topk(acts, k=4)
    assert (out > 0).sum() == 4 * 8


def test_batch_topk_shares_budget_across_samples():
    # One row dominates; BatchTopK should spend the budget there rather than
    # forcing each row to keep the same number of features.
    acts = torch.zeros(2, 4)
    acts[0] = torch.tensor([9.0, 8.0, 7.0, 6.0])
    acts[1] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = batch_topk(acts, k=2)
    assert (out[0] > 0).sum() == 4
    assert (out[1] > 0).sum() == 0


def test_batch_topk_zero_k():
    assert batch_topk(torch.rand(4, 8), k=0).sum() == 0


def test_dfc_partition_sizes():
    cfg = CrosscoderConfig(d_a=32, d_b=32, n_features=1000, exclusive_frac=0.05)
    assert cfg.n_exclusive == 50
    assert cfg.n_shared == 900
    assert cfg.is_dfc


def test_standard_crosscoder_has_no_exclusive_partition():
    cfg = CrosscoderConfig(d_a=32, d_b=32, n_features=1000, exclusive_frac=0.0)
    assert not cfg.is_dfc
    assert cfg.n_shared == 1000
    model = Crosscoder(cfg)
    # Both decoders span the whole dictionary.
    assert model.W_dec_a.shape[0] == 1000
    assert model.W_dec_b.shape[0] == 1000


def test_invalid_exclusive_frac():
    with pytest.raises(ValueError):
        CrosscoderConfig(d_a=8, d_b=8, exclusive_frac=0.6)


def test_dfc_decoder_norms_are_structurally_zero():
    """The DFC's core guarantee: exclusivity is exact, not approximate."""
    cfg = CrosscoderConfig(d_a=16, d_b=24, n_features=200, exclusive_frac=0.1, k=4)
    model = Crosscoder(cfg)
    norms_a, norms_b = model.decoder_norms()

    a_lo, a_hi = model.idx_excl_a
    b_lo, b_hi = model.idx_excl_b
    assert norms_b[a_lo:a_hi].abs().max() == 0.0, "I_A features must not reconstruct B"
    assert norms_a[b_lo:b_hi].abs().max() == 0.0, "I_B features must not reconstruct A"
    # Shared features reconstruct both.
    s_lo, s_hi = model.idx_shared
    assert norms_a[s_lo:s_hi].min() > 0
    assert norms_b[s_lo:s_hi].min() > 0


def test_dfc_exclusive_features_get_no_gradient_from_other_model():
    """Severed decoder weights mean no gradient path, which is the mechanism
    that removes the pressure to become shared."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=200, exclusive_frac=0.1, k=4)
    model = Crosscoder(cfg)
    x_a = torch.randn(32, 16)
    x_b = torch.randn(32, 16)

    out = model.forward(x_a, x_b)
    # Only model B's reconstruction error.
    loss_b = torch.nn.functional.mse_loss(out["recon_b"], x_b)
    loss_b.backward()

    a_lo, a_hi = model.idx_excl_a
    grad = model.W_enc_a.grad[:, a_lo:a_hi]
    assert grad.abs().max() == 0.0


def test_relative_decoder_norm_ranges():
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=200, exclusive_frac=0.1, k=4)
    model = Crosscoder(cfg)
    rel = model.relative_decoder_norm()
    a_lo, a_hi = model.idx_excl_a
    b_lo, b_hi = model.idx_excl_b
    assert torch.allclose(rel[a_lo:a_hi], torch.ones(a_hi - a_lo), atol=1e-4)
    assert torch.allclose(rel[b_lo:b_hi], torch.zeros(b_hi - b_lo), atol=1e-4)


def test_partition_of_and_mask_agree():
    cfg = CrosscoderConfig(d_a=8, d_b=8, n_features=100, exclusive_frac=0.1, k=2)
    model = Crosscoder(cfg)
    for name in ("A", "S", "B"):
        mask = model.partition_mask(name)
        for idx in mask.nonzero(as_tuple=True)[0].tolist():
            assert model.partition_of(idx) == name


def test_sparsity_annealing_decays_k():
    cfg = CrosscoderConfig(d_a=8, d_b=8, n_features=100, k=10, k_initial=50, anneal_steps=100)
    model = Crosscoder(cfg)
    assert model.current_k(0) == 50
    assert model.current_k(100) == 10
    assert model.current_k(500) == 10
    assert 10 < model.current_k(50) < 50


def test_dead_feature_tracking():
    cfg = CrosscoderConfig(d_a=8, d_b=8, n_features=50, k=2, dead_after_tokens=100)
    model = Crosscoder(cfg)
    feats = torch.zeros(64, 50)
    feats[:, 0] = 1.0
    model.update_dead_features(feats)
    assert model.tokens_since_fired[0] == 0
    model.update_dead_features(feats)
    assert model.dead_mask()[1], "a feature that never fires should die"
    assert not model.dead_mask()[0]


def test_fve_perfect_and_zero():
    x = torch.randn(64, 8)
    assert fraction_variance_explained(x, x) > 0.999
    mean_only = x.mean(dim=0, keepdim=True).expand_as(x)
    assert abs(float(fraction_variance_explained(x, mean_only))) < 1e-3
