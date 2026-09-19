import pytest
import torch

from diffdiff.data.toy import ToyActivations, cross_arch_config, null_config
from diffdiff.diffing.crosscoder import Crosscoder, CrosscoderConfig
from diffdiff.diffing.train import TrainConfig, collect_features, mask_outliers, train
from diffdiff.validate.mirror import find_mirror_pairs
from diffdiff.validate.null import run_null_test

SMALL_CC = dict(n_features=256, exclusive_frac=0.05, k=8, k_initial=16, anneal_steps=20)
SMALL_TRAIN = dict(steps=40, batch_size=64, log_every=20)


def test_null_test_rejects_non_null_config():
    with pytest.raises(ValueError, match="null configuration"):
        run_null_test(toy=cross_arch_config())


def test_null_test_runs_end_to_end():
    report = run_null_test(
        toy=null_config(d=32, n_shared=64),
        crosscoder=CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC),
        training=TrainConfig(**SMALL_TRAIN),
        eval_batches=2,
        compare_standard=False,
    )
    assert 0.0 <= report.exclusive_mass_frac <= 1.0
    assert 0.0 <= report.exclusive_live_frac <= 1.0
    assert report.verdict()
    assert "exclusive mass" in report.summary()


def test_mirror_requires_dedicated_partitions():
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=128, exclusive_frac=0.0, k=4)
    model = Crosscoder(cfg)
    with pytest.raises(ValueError, match="dedicated partitions"):
        find_mirror_pairs(model, torch.rand(32, 128))


def test_mirror_report_handles_empty_partitions():
    """A partition that never fires is a clean result, not a crash."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=128, exclusive_frac=0.05, k=4)
    model = Crosscoder(cfg)
    feats = torch.zeros(32, 128)
    feats[:, model.idx_shared[0] : model.idx_shared[0] + 4] = torch.rand(32, 4)
    report = find_mirror_pairs(model, feats)
    assert report.n_live_a == 0
    assert report.mirror_frac == 0.0
    assert report.pairs == []


def test_mirror_detects_planted_duplicate():
    """Two exclusive features driven by the same signal must be found."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=128, exclusive_frac=0.05, k=4)
    model = Crosscoder(cfg)
    n_samples = 512
    feats = torch.rand(n_samples, 128) * 0.01
    signal = torch.rand(n_samples)
    a_lo, _ = model.idx_excl_a
    b_lo, _ = model.idx_excl_b
    feats[:, a_lo] = signal
    feats[:, b_lo] = signal  # a perfect mirror pair

    report = find_mirror_pairs(model, feats, threshold=0.9)
    assert (a_lo, b_lo, pytest.approx(1.0, abs=1e-3)) in [
        (a, b, pytest.approx(s, abs=1e-3)) for a, b, s in report.pairs
    ]


def test_mirror_control_uses_shared_partition():
    """The control must be computed, so mirror_frac is never read raw."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=256, exclusive_frac=0.05, k=4)
    model = Crosscoder(cfg)
    feats = torch.rand(256, 256)
    report = find_mirror_pairs(model, feats)
    assert report.excess == pytest.approx(report.mirror_frac - report.control_frac)


def test_mask_outliers_drops_large_norms():
    x_a = torch.ones(10, 4)
    x_a[0] *= 100.0
    x_b = torch.ones(10, 4)
    kept_a, kept_b = mask_outliers(x_a, x_b, mult=2.0)
    assert kept_a.shape[0] == 9
    assert kept_b.shape[0] == 9


def test_mask_outliers_never_empties_batch():
    x = torch.ones(4, 2)
    kept_a, _ = mask_outliers(x * 0, x * 0, mult=2.0)
    assert kept_a.shape[0] == 4


def test_training_improves_reconstruction():
    src = ToyActivations(null_config(d=32, n_shared=64))
    model = Crosscoder(CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC))
    history = train(
        model, src.sample_activations,
        TrainConfig(steps=200, batch_size=128, lr=1e-3, log_every=50),
    )
    assert history[-1]["fve_a"] > history[0]["fve_a"]


def test_collect_features_shape():
    src = ToyActivations(null_config(d=32, n_shared=64))
    model = Crosscoder(CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC))
    feats = collect_features(model, src.sample_activations, n_batches=3, batch_size=16)
    assert feats.shape == (48, 256)


def test_control_rejects_config_without_differences():
    from diffdiff.data.toy import null_config as _null
    from diffdiff.validate.null import run_control_test

    with pytest.raises(ValueError, match="exclusive or dropped concepts"):
        run_control_test(_null(d=16, n_shared=32))


def test_control_runs_on_cross_arch():
    from diffdiff.validate.null import run_control_test

    report = run_control_test(
        toy=cross_arch_config(d_a=32, d_b=32, n_shared=56, n_excl=8),
        crosscoder=CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC),
        training=TrainConfig(**SMALL_TRAIN),
        eval_batches=2,
        compare_standard=False,
    )
    assert 0.0 <= report.exclusive_mass_frac <= 1.0


def test_regime_test_allows_non_null():
    """The generalised runner must not carry the null guard."""
    from diffdiff.validate.null import run_regime_test

    report = run_regime_test(
        toy=cross_arch_config(d_a=32, d_b=32, n_shared=56, n_excl=8),
        crosscoder=CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC),
        training=TrainConfig(**SMALL_TRAIN),
        eval_batches=2,
        compare_standard=False,
    )
    assert report.verdict()


def test_mirror_control_pools_match_candidate_counts():
    """The control's max is over the same number of candidates as the real
    test, since a maximum grows with pool size."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=512, exclusive_frac=0.05, k=4)
    model = Crosscoder(cfg)
    feats = torch.rand(256, 512)
    report = find_mirror_pairs(model, feats)
    assert report.control_matched, "shared partition is large enough here"


def test_mirror_control_flags_undersized_pool():
    """A shared partition too small for matched pools must be flagged, not
    silently produce an optimistic excess."""
    cfg = CrosscoderConfig(d_a=16, d_b=16, n_features=100, exclusive_frac=0.4, k=4)
    model = Crosscoder(cfg)
    feats = torch.rand(128, 100)
    report = find_mirror_pairs(model, feats)
    assert not report.control_matched
    assert "[control pool undersized]" in report.summary()


def test_control_verdict_does_not_reuse_null_thresholds():
    """A control run's exclusive partitions are supposed to be populated, so
    the null's pass/fail language would invert the result."""
    from diffdiff.validate.null import run_control_test

    report = run_control_test(
        toy=cross_arch_config(d_a=32, d_b=32, n_shared=56, n_excl=8),
        crosscoder=CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC),
        training=TrainConfig(**SMALL_TRAIN),
        eval_batches=2,
        compare_standard=False,
    )
    assert not report.is_null
    assert report.verdict().startswith("CONTROL")
    assert "FAIL" not in report.verdict()


def test_null_report_defaults_to_null_semantics():
    from diffdiff.validate.null import run_null_test

    report = run_null_test(
        toy=null_config(d=32, n_shared=64),
        crosscoder=CrosscoderConfig(d_a=32, d_b=32, **SMALL_CC),
        training=TrainConfig(**SMALL_TRAIN),
        eval_batches=2,
        compare_standard=False,
    )
    assert report.is_null
    assert not report.verdict().startswith("CONTROL")
