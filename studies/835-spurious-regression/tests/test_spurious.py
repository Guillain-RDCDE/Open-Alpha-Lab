"""Offline, fixed-seed tests for the spurious-regression machinery.

The pitfall appears at the planted magnitude (level OLS on two independent random walks
over-rejects massively and prints high R²); the fix removes it (first-differencing
restores ~5% size); the pitfall is SPECIFIC to nonstationarity (stationary series are
correctly sized); the cointegration test tells a genuine relation from a spurious one;
more data makes the level test worse; the pairs trade has no tradable edge; and the
inference primitives behave. All deterministic and offline.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spurious_regression import data, strategy as st  # noqa: E402


# --- determinism ------------------------------------------------------------ #
def test_worlds_deterministic():
    a = data.independent_walks(50, n_obs=200, seed=835)
    b = data.independent_walks(50, n_obs=200, seed=835)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
    c = data.cointegrated_pairs(20, n_obs=200, seed=835)
    d = data.cointegrated_pairs(20, n_obs=200, seed=835)
    assert np.allclose(c[0], d[0]) and np.allclose(c[1], d[1])


def test_independent_walks_are_i1_and_unrelated():
    X, Y = data.independent_walks(500, n_obs=250, seed=835)
    # each series is a random walk: its first difference is ~white noise (low autocorr)
    dx = np.diff(X, axis=1)
    ac1 = np.mean([np.corrcoef(dx[i, :-1], dx[i, 1:])[0, 1] for i in range(200)])
    assert abs(ac1) < 0.1
    # the two shock streams are independent: mean cross-correlation of increments ~ 0
    dy = np.diff(Y, axis=1)
    xc = np.mean([np.corrcoef(dx[i], dy[i])[0, 1] for i in range(200)])
    assert abs(xc) < 0.05


# --- the pitfall: level OLS on random walks over-rejects --------------------- #
def test_level_ols_grossly_oversized(walks):
    ex = st.regression_experiment(*walks)["level"]
    # a valid 5% test rejects ~5%; the spurious level regression rejects the vast majority
    assert ex["reject_rate"] > 0.7
    assert ex["mean_abs_t"] > 5.0          # the average |t| is enormous
    assert ex["mean_r2"] > 0.10            # and R² is high for a pure null


def test_first_difference_restores_correct_size(walks):
    ex = st.regression_experiment(*walks)
    lvl, dif = ex["level"], ex["diff"]
    # the FIX: differencing collapses the rejection rate back to ~nominal 5%
    assert dif["reject_rate"] < 0.08
    assert dif["mean_r2"] < 0.02
    assert dif["reject_rate"] < lvl["reject_rate"] / 5.0   # a >5x correction


def test_trending_series_make_it_worse():
    Xd, Yd = data.independent_walks(1500, n_obs=250, drift=0.15, seed=835)
    ex = st.regression_experiment(Xd, Yd)["level"]
    # a shared deterministic trend pushes the spurious rejection rate toward 1 and R² high
    assert ex["reject_rate"] > 0.95
    assert ex["mean_r2"] > 0.4


# --- specificity: the pitfall is nonstationarity, not OLS -------------------- #
def test_stationary_series_correctly_sized(stationary):
    ex = st.regression_experiment(*stationary)["level"]
    assert ex["reject_rate"] < 0.08        # ~5% — OLS itself is fine
    assert ex["mean_r2"] < 0.02


# --- more data makes the LEVEL test worse, the DIFF test stays correct ------- #
def test_reject_rate_rises_with_sample_size():
    rows = st.sample_size_sweep(data, n_obs_grid=(50, 250, 1000), n_pairs=2000, seed_base=835)
    lvl = [r["level_reject"] for r in rows]
    dif = [r["diff_reject"] for r in rows]
    assert lvl[0] < lvl[-1]                 # level rejection rate GROWS with n
    assert max(dif) < 0.09                  # differenced rejection stays near nominal


# --- the cointegration test tells spurious from genuine --------------------- #
def test_cointegration_distinguishes_spurious_from_genuine(cointegrated):
    Xi, Yi = data.independent_walks(120, n_obs=250, seed=835)
    ci = st.cointegration_reject_rate(Xi, Yi)
    cc = st.cointegration_reject_rate(*cointegrated)
    assert ci["reject_rate"] < 0.15         # independent walks: correctly finds ~nothing
    assert cc["reject_rate"] > 0.80         # genuine cointegration: correctly detected


# --- tradability: the spurious spread is not tradable ----------------------- #
def test_pairs_trade_has_no_significant_edge_and_costs_hurt(walks):
    gross = st.pairs_timer(*walks, cost_bps=0.0, borrow_bps_yr=0.0)
    net = st.pairs_timer(*walks, cost_bps=5.0, borrow_bps_yr=50.0)
    assert abs(gross["t_net"]) < 2.0        # no edge distinguishable from zero
    assert net["net_bps"] < gross["gross_bps"]   # costs only make it worse


# --- OLS batch matches a direct polyfit ------------------------------------- #
def test_ols_batch_matches_polyfit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 80)); Y = 0.7 * X + rng.normal(size=(5, 80))
    res = st.ols_batch(X, Y)
    for i in range(5):
        b, a = np.polyfit(X[i], Y[i], 1)
        assert abs(res["beta"][i] - b) < 1e-9
        # R² cross-check
        yhat = a + b * X[i]
        r2 = 1 - np.sum((Y[i] - yhat) ** 2) / np.sum((Y[i] - Y[i].mean()) ** 2)
        assert abs(res["r2"][i] - r2) < 1e-9


# --- inference primitives ---------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(85, 100)
    assert lo < 0.85 < hi


def test_welch_t_zero_on_equal_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 2000); b = rng.normal(0, 1, 2000)
    assert abs(st.welch_t(a, b)) < 3.0
