"""The synthetic battery is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiple_testing import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
BATTERY_CACHE = os.path.join(CACHE, "battery_SPY.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_battery):
    rets, truth, meta = null_battery
    assert rets.shape == (meta["n_days"], meta["m_effects"])
    assert len(truth) == meta["m_effects"]
    assert list(rets.columns) == list(truth.index)
    assert np.isfinite(rets.to_numpy()).all()


def test_null_truth_is_all_false(null_battery):
    _, truth, _ = null_battery
    assert truth.sum() == 0


def test_planted_truth_count(strong_battery):
    _, truth, meta = strong_battery
    assert truth.sum() == meta["n_true"] == 10
    # the first n_true columns are the planted ones
    assert truth.iloc[:10].all()
    assert not truth.iloc[10:].any()


def test_synthetic_is_deterministic():
    a, ta, _ = data.synthetic_battery(n_days=1000, m_effects=40, n_true=4, seed=7)
    b, tb, _ = data.synthetic_battery(n_days=1000, m_effects=40, n_true=4, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert (ta == tb).all()
    c, _, _ = data.synthetic_battery(n_days=1000, m_effects=40, n_true=4, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_planted_effects_have_higher_mean():
    """The planted effects earn a positive mean on their active days; nulls average ~0."""
    rets, truth, _ = data.synthetic_battery(n_days=5000, m_effects=60, n_true=6,
                                            edge=0.004, seed=346)
    real_cols = truth[truth].index
    null_cols = truth[~truth].index
    real_mean = rets[real_cols].mean().mean()
    null_mean = rets[null_cols].mean().mean()
    assert real_mean > null_mean
    assert real_mean > 0


def test_index_uses_safe_range_no_overflow():
    rets, _, _ = data.synthetic_battery(n_days=2520, m_effects=10, seed=1)
    assert rets.index.is_monotonic_increasing
    assert rets.index[0].year >= 1900
    assert rets.index[-1].year < 2300


def test_fingerprint_stable_and_sensitive(null_battery):
    rets, _, _ = null_battery
    assert data.fingerprint(rets) == data.fingerprint(rets)
    other, _, _ = data.synthetic_battery(n_days=2520, m_effects=100, n_true=0, seed=99)
    assert data.fingerprint(rets) != data.fingerprint(other)


def test_effect_masks_are_calendar_known():
    """Calendar masks are deterministic functions of the index — no look-ahead, no noise."""
    idx = data.pd.date_range("2010-01-01", periods=500, freq="B")
    m1 = data._effect_masks(idx)
    m2 = data._effect_masks(idx)
    assert set(m1) >= set(data.CALENDAR_EFFECTS)
    for name in data.CALENDAR_EFFECTS:
        assert np.array_equal(m1[name], m2[name])
    # mutually-exclusive day-of-week dummies cover every business day exactly once
    dow_union = np.zeros(len(idx), dtype=bool)
    for d in ["mon", "tue", "wed", "thu", "fri"]:
        dow_union |= m1[d]
    assert dow_union.all()


# ---- real loader: cache contract -------------------------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(BATTERY_CACHE), reason="offline CI: no real cache")
def test_real_battery_loads_from_cache():
    rets, truth = data.load_real(fetch=False)
    assert rets.shape[1] == len(data.CALENDAR_EFFECTS)
    assert list(rets.columns) == list(truth.index)
    # the real truth mask is all-False (we don't know which calendar effects are real)
    assert truth.sum() == 0
    assert np.isfinite(rets.to_numpy()).all()
