"""The synthetic worlds are deterministic, offline, and carry the ground truth the demo needs:
the nulls have NO point-in-time-tradeable edge (so any backtest 'edge' is an artefact), and the
planted world has a genuine one."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lookahead_standardization import data  # noqa: E402


def test_worlds_deterministic():
    for fn in (data.null_stationary, data.null_nonstationary, data.planted_edge):
        a = fn(seed=data.BASE_SEED)
        b = fn(seed=data.BASE_SEED)
        assert np.allclose(a[0], b[0]) and np.allclose(np.nan_to_num(a[1]), np.nan_to_num(b[1]))


def test_shapes_and_forward_nans():
    X, R = data.null_nonstationary(seed=1, n_names=60, n_days=1000, horizon=10)
    assert X.shape == (1000, 60) and R.shape == (1000, 60)
    # the last `horizon` rows have no forward return (NaN) — no peeking past the end
    assert np.isnan(R[-1]).all() and np.isnan(R[-10:]).all()
    assert np.isfinite(R[:-10]).all()


def test_random_walk_is_nonstationary():
    """The trap feature is a genuine random walk: its variance grows ~linearly with time."""
    X, _ = data.null_nonstationary(seed=2, n_names=200, n_days=1000)
    v_early = X[100].var()
    v_late = X[900].var()
    assert v_late > 4 * v_early          # variance of a RW grows with t (non-stationary)


def test_stationary_feature_is_stationary():
    X, _ = data.null_stationary(seed=2, n_names=200, n_days=1000, phi=0.9)
    v_early = X[100].var()
    v_late = X[900].var()
    assert 0.5 < v_late / v_early < 2.0  # variance roughly constant (stationary)


def test_nonstationary_returns_unpredictable_from_past():
    """The forward return is the RW's future change — iid increments make it independent of the past
    LEVEL, so a point-in-time signal genuinely cannot forecast it (the null is real)."""
    X, R = data.null_nonstationary(seed=3, n_names=400, n_days=800, horizon=10)
    lvl = X[300]                         # feature level known at t=300
    fwd = R[300]                         # forward return over (300, 310]
    m = np.isfinite(lvl) & np.isfinite(fwd)
    corr = np.corrcoef(lvl[m], fwd[m])[0, 1]
    assert abs(corr) < 0.15              # current level does not predict the future change


def test_planted_edge_is_real_and_pointintime():
    """In the control the CURRENT feature genuinely predicts the next return (a real edge)."""
    X, R = data.planted_edge(seed=3, n_names=400, n_days=800, beta=0.10)
    lvl = X[300]
    fwd = R[300]
    m = np.isfinite(lvl) & np.isfinite(fwd)
    corr = np.corrcoef(lvl[m], fwd[m])[0, 1]
    assert corr > 0.03                   # positive, real, available in real time


def test_config_fingerprint_stable():
    assert data.config_fingerprint(n_seeds=20) == data.config_fingerprint(n_seeds=20)
    assert data.config_fingerprint(n_seeds=20) != data.config_fingerprint(n_seeds=25)


def test_scale_invariance_of_daily_vol():
    """Scaling the RW increment vol rescales prices/returns but leaves the leak (scale-free) intact."""
    from lookahead_standardization import strategy as st
    Xa, Ra = data.null_nonstationary(seed=5, daily_vol=0.01)
    Xb, Rb = data.null_nonstationary(seed=5, daily_vol=0.05)
    ic_a = np.nanmean(st.cross_sectional_ic(st.full_standardize(Xa), Ra))
    ic_b = np.nanmean(st.cross_sectional_ic(st.full_standardize(Xb), Rb))
    assert abs(ic_a - ic_b) < 1e-9       # IC is scale-invariant
