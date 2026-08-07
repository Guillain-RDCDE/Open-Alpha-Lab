"""The study's spine: (1) expanding standardisation is genuinely point-in-time (blind to the future)
while full-sample standardisation is not; (2) on the non-stationary null the leaky (full-sample)
z-score manufactures a large fake IC/Sharpe while the honest (expanding) one reads ~0; (3) on the
stationary null neither leaks; (4) on a planted real edge the honest method RECOVERS it (unbiased,
not always-zero). Plus the inference primitives and the leak-magnitude scaling laws."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lookahead_standardization import data, strategy as st  # noqa: E402


# ---- the two standardisations: one leaks the future, one does not ----------
def test_expanding_is_point_in_time():
    """Perturbing FUTURE rows must not change an expanding z-score at earlier rows (no look-ahead).
    The same perturbation MUST move the full-sample z-score (it peeks at the future)."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.normal(0, 1, (300, 5)), axis=0)
    Ze = st.expanding_standardize(X, min_periods=60)
    Zf = st.full_standardize(X)
    X2 = X.copy()
    X2[200:] += 7.0                       # change only the FUTURE (rows >= 200)
    Ze2 = st.expanding_standardize(X2, min_periods=60)
    Zf2 = st.full_standardize(X2)
    # expanding: rows before the perturbation are untouched
    assert np.allclose(Ze[60:200], Ze2[60:200], equal_nan=True)
    # full-sample: even an early row's z changed, because the mean/std saw the future
    assert not np.allclose(np.nan_to_num(Zf[100]), np.nan_to_num(Zf2[100]))


def test_full_standardize_is_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    X = rng.normal(3.0, 2.0, (500, 8))
    Z = st.full_standardize(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-9)
    assert np.allclose(Z.std(0), 1, atol=1e-9)


# ---- (2) the trap: full-sample leaks on the non-stationary null ------------
def test_nonstationary_null_full_leaks_expanding_clean(nonstationary_null):
    X, R = nonstationary_null
    ic_f = st.cross_sectional_ic(st.full_standardize(X), R)
    ic_e = st.cross_sectional_ic(st.expanding_standardize(X, 60), R)
    tf = st.newey_west_t(ic_f)
    te = st.newey_west_t(ic_e)
    assert abs(np.nanmean(ic_f)) > 0.08   # full-sample manufactures a large IC out of noise
    assert abs(tf) > 5                     # ...and it is wildly "significant"
    assert abs(np.nanmean(ic_e)) < 0.02    # expanding (honest) reads ~0
    assert abs(te) < 2                     # ...and is not significant


# ---- (3) the contrast: stationary null leaks (almost) nothing --------------
def test_stationary_null_no_leak(stationary_null):
    X, R = stationary_null
    ic_f = st.cross_sectional_ic(st.full_standardize(X), R)
    ic_e = st.cross_sectional_ic(st.expanding_standardize(X, 60), R)
    assert abs(np.nanmean(ic_f)) < 0.02    # a stationary feature's full-sample z barely leaks
    assert abs(np.nanmean(ic_e)) < 0.02


# ---- (4) the control: the honest method recovers a REAL edge ---------------
def test_planted_edge_recovered_by_expanding(planted):
    X, R = planted
    ic_e = st.cross_sectional_ic(st.expanding_standardize(X, 60), R)
    assert np.nanmean(ic_e) > 0.03         # honest method finds the real edge
    assert st.newey_west_t(ic_e) > 5       # strongly significant, in the RIGHT direction


# ---- the leak_report gap and the seed-robust machinery-proof ----------------
def test_leak_report_gap_positive_on_trap(nonstationary_null):
    X, R = nonstationary_null
    rep = st.leak_report(X, R)
    assert rep["abs_ic_gap"] > 0.08        # full-sample |IC| dwarfs expanding |IC|
    assert rep["sharpe_gap"] > 5           # and the fake Sharpe gap is huge


def test_seed_robust_full_fires_expanding_silent():
    r = st.seed_robust(data.null_nonstationary, n_seeds=20, base_seed=data.BASE_SEED)
    assert r["full_sig_seeds"] >= 18       # leak fires on ~every seed
    assert r["exp_sig_seeds"] <= 2         # honest method silent on ~every seed


def test_seed_robust_expanding_fires_on_planted():
    r = st.seed_robust(data.planted_edge, n_seeds=20, base_seed=data.BASE_SEED)
    assert r["exp_sig_seeds"] >= 18        # the honest method is NOT always-zero — it finds real edge


# ---- the leak-magnitude scaling laws (the finite-sample fingerprint) --------
def test_leak_grows_with_horizon():
    hs = st.horizon_sweep(horizons=(1, 10, 40), n_seeds=6)
    ics = hs["full_ic"].abs().to_numpy()
    assert ics[0] < ics[1] < ics[2]        # longer horizon -> bigger leak


def test_leak_dilutes_with_sample_length():
    ls = st.length_sweep(lengths=(250, 1000, 2000), n_seeds=6)
    ics = ls["full_ic"].abs().to_numpy()
    assert ics[0] > ics[1] > ics[2]        # longer sample -> smaller leak (finite-sample artefact)


# ---- the costed timer ------------------------------------------------------
def test_costs_reduce_net(nonstationary_null):
    X, R = nonstationary_null
    sp = st.long_short_spread(st.full_standardize(X), R, frac=0.2)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


# ---- inference primitives --------------------------------------------------
def test_spearman_matches_known_sign():
    a = np.array([1.0, 2, 3, 4, 5])
    assert st._spearman(a, a) > 0.99
    assert st._spearman(a, -a) < -0.99


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert st.welch_t(a, b) > 2


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi
