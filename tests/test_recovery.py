import pytest
import torch

from diffdiff.data.toy import ToyActivations, cross_arch_config, null_config, quantized_config
from diffdiff.diffing.crosscoder import Crosscoder, CrosscoderConfig
from diffdiff.validate.recovery import concept_recovery


def make(d_a=32, d_b=24, n_shared=48, n_excl=8, frac=0.05, n_features=200):
    src = ToyActivations(cross_arch_config(d_a=d_a, d_b=d_b, n_shared=n_shared, n_excl=n_excl))
    model = Crosscoder(CrosscoderConfig(d_a=d_a, d_b=d_b, n_features=n_features,
                                        exclusive_frac=frac, k=4))
    return src, model


def test_rejects_bad_model_arg():
    src, model = make()
    with pytest.raises(ValueError, match="'a' or 'b'"):
        concept_recovery(model, src, which="c")


def test_report_fields_in_range():
    src, model = make()
    r = concept_recovery(model, src, which="a")
    for v in (r.recall, r.precision, r.f1, r.shared_leakage, r.unmatched):
        assert 0.0 <= v <= 1.0
    assert r.n_true == 8
    assert r.n_labelled == 10  # 5% of 200


def test_budget_is_matched_across_architectures():
    """A DFC and a standard crosscoder must label the same number of features
    exclusive, or recall is not comparable between them."""
    src, dfc = make(frac=0.05)
    _, std = make(frac=0.0)
    ra = concept_recovery(dfc, src, which="a")
    rb = concept_recovery(std, src, which="a")
    assert ra.n_labelled == rb.n_labelled


def test_planted_exclusive_concept_is_recovered():
    """A feature whose decoder IS a true exclusive concept must be counted."""
    d = 32
    src = ToyActivations(cross_arch_config(d_a=d, d_b=d, n_shared=48, n_excl=8))
    cfg = CrosscoderConfig(d_a=d, d_b=d, n_features=200, exclusive_frac=0.05, k=4)
    model = Crosscoder(cfg)

    gt = src.ground_truth()
    true_excl = (gt["visible_a"] & ~gt["visible_b"]).nonzero(as_tuple=True)[0]
    lo, hi = model.idx_excl_a
    with torch.no_grad():
        # Plant real exclusive concepts into the A-exclusive partition.
        for slot, concept in enumerate(true_excl[: hi - lo]):
            model.W_dec_a[lo + slot] = src.concepts_a[concept]

    r = concept_recovery(model, src, which="a", threshold=0.9)
    assert r.recall > 0.9, "planted exclusive concepts should be recovered"
    assert r.precision > 0.7


def test_shared_leakage_detected():
    """Labelling shared concepts as exclusive must show up as leakage, since
    that is the false positive a user would chase and find spurious."""
    d = 32
    src = ToyActivations(cross_arch_config(d_a=d, d_b=d, n_shared=48, n_excl=8))
    model = Crosscoder(CrosscoderConfig(d_a=d, d_b=d, n_features=200,
                                        exclusive_frac=0.05, k=4))
    gt = src.ground_truth()
    shared = (gt["visible_a"] & gt["visible_b"]).nonzero(as_tuple=True)[0]
    lo, hi = model.idx_excl_a
    with torch.no_grad():
        for slot, concept in enumerate(shared[: hi - lo]):
            model.W_dec_a[lo + slot] = src.concepts_a[concept]

    r = concept_recovery(model, src, which="a", threshold=0.9)
    assert r.shared_leakage > 0.9
    assert r.precision < 0.1


def test_unmatched_counted_for_random_decoders():
    """Untrained random decoders should match nothing, not score precision."""
    src, model = make(d_a=64, d_b=64, n_shared=200, n_excl=8)
    r = concept_recovery(model, src, which="a", threshold=0.8)
    assert r.unmatched > 0.8
    assert r.precision < 0.2


def test_dropped_concepts_count_as_exclusive_to_a():
    """A concept destroyed in B is exclusive to A, and must be scoreable."""
    d = 32
    src = ToyActivations(quantized_config(d=d, n_shared=64, drop_concepts=(1, 2, 3)))
    model = Crosscoder(CrosscoderConfig(d_a=d, d_b=d, n_features=200,
                                        exclusive_frac=0.05, k=4))
    r = concept_recovery(model, src, which="a")
    assert r.n_true == 3


def test_null_has_no_true_exclusive_concepts():
    d = 32
    src = ToyActivations(null_config(d=d, n_shared=64))
    model = Crosscoder(CrosscoderConfig(d_a=d, d_b=d, n_features=200,
                                        exclusive_frac=0.05, k=4))
    r = concept_recovery(model, src, which="a")
    assert r.n_true == 0
    assert r.recall == 0.0
