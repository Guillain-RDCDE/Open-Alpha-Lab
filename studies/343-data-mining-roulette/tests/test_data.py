"""The synthetic tape is well-formed, deterministic, and respects the no-look-ahead
contract; the real loader is cache- and offline-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_mining_roulette import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
REAL_CACHE = os.path.join(CACHE, "roulette_SPY.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_tape):
    feats, fwd, truth = null_tape
    assert feats.shape == (truth["n_days"], truth["n_features"])
    assert len(fwd) == truth["n_days"]
    assert list(feats.columns) == data.FEATURE_NAMES
    assert feats.columns[0] == "f0_signal"


def test_features_finite_after_warmup(null_tape):
    feats, _, _ = null_tape
    # The first ~50 rows have NaNs from rolling windows; everything after must be finite.
    tail = feats.iloc[60:]
    assert np.isfinite(tail.to_numpy()).all()


def test_synthetic_is_deterministic():
    a_f, a_r, _ = data.synthetic_tape(n_days=500, planted_edge=0.05, seed=7)
    b_f, b_r, _ = data.synthetic_tape(n_days=500, planted_edge=0.05, seed=7)
    assert np.allclose(a_f.to_numpy(), b_f.to_numpy(), equal_nan=True)
    assert np.allclose(a_r.to_numpy(), b_r.to_numpy(), equal_nan=True)
    c_f, _, _ = data.synthetic_tape(n_days=500, planted_edge=0.05, seed=8)
    assert not np.allclose(a_f.to_numpy(), c_f.to_numpy(), equal_nan=True)


def test_null_signal_feature_is_uncorrelated_with_forward():
    """At planted_edge=0 the signal feature does NOT predict tomorrow (a clean null)."""
    feats, fwd, _ = data.synthetic_tape(n_days=8000, planted_edge=0.0, seed=343)
    aligned = feats["f0_signal"].iloc[:-1].to_numpy()
    target = fwd.iloc[:-1].to_numpy()
    corr = np.corrcoef(aligned, target)[0, 1]
    assert abs(corr) < 0.05


def test_planted_edge_makes_signal_feature_predictive():
    """At planted_edge>0 the signal feature DOES predict tomorrow's return."""
    feats, fwd, _ = data.synthetic_tape(n_days=8000, planted_edge=0.10, seed=343)
    aligned = feats["f0_signal"].iloc[:-1].to_numpy()
    target = fwd.iloc[:-1].to_numpy()
    corr = np.corrcoef(aligned, target)[0, 1]
    assert corr > 0.04


def test_decoys_stay_noise_under_planted_edge():
    """A decoy feature must NOT become predictive when only f0 carries the edge."""
    feats, fwd, _ = data.synthetic_tape(n_days=8000, planted_edge=0.10, seed=343)
    aligned = feats["noise_a"].iloc[:-1].to_numpy()
    target = fwd.iloc[:-1].to_numpy()
    corr = np.corrcoef(aligned, target)[0, 1]
    assert abs(corr) < 0.05


def test_index_is_safe_no_overflow(null_tape):
    """Decorative date labels are built without overflowing ns-Timestamps."""
    feats, _, _ = null_tape
    assert feats.index.is_monotonic_increasing
    assert 1900 <= feats.index[0].year <= 2100
    assert feats.index[-1].year < 2100


def test_fingerprint_stable_and_sensitive(null_tape):
    feats, fwd, _ = null_tape
    assert data.fingerprint(feats, fwd) == data.fingerprint(feats, fwd)
    other_f, other_r, _ = data.synthetic_tape(n_days=2520, planted_edge=0.0, seed=99)
    assert data.fingerprint(feats, fwd) != data.fingerprint(other_f, other_r)


# ---- real loader: cache contract -------------------------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="offline CI: no real cache")
def test_real_tape_loads_from_cache():
    feats, fwd = data.load_real(fetch=False)
    assert list(feats.columns) == data.FEATURE_NAMES
    assert len(feats) == len(fwd)
    assert feats.index.equals(fwd.index)
