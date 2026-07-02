"""The engine and the study's spine: (1) smoothing inflates the NAIVE Sharpe while leaving the
HONEST (autocorrelation-corrected) Sharpe unchanged — a pure measurement game; (2) naive leverage
changes NEITHER Sharpe; (3) the honest metric tracks REAL edge (the control). Plus the honesty
disciplines: smoothing preserves the mean, the honest correction equals naive under iid, costs never
raise the Sharpe."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sharpe_hacking import data  # noqa: E402
from sharpe_hacking import strategy as st  # noqa: E402


# ---- the two Sharpe measurements ------------------------------------------
def test_sharpe_zero_on_degenerate():
    assert st.sharpe_naive(np.zeros(100)) == 0.0
    assert st.sharpe_naive(np.array([0.01])) == 0.0
    assert st.sharpe_honest_annual(np.zeros(100)) == 0.0


def test_honest_equals_naive_under_iid():
    """For an iid series the autocorrelation correction is a no-op: honest ~ naive."""
    rng = np.random.default_rng(0)
    r = 0.0004 + 0.01 * rng.standard_normal(4000)
    assert abs(st.sharpe_honest_annual(r) - st.sharpe_naive(r)) < 0.05


# ---- lever 1: SMOOTHING inflates the naive Sharpe, not the honest one -------
def test_smoothing_preserves_mean(null_tape):
    r = null_tape.to_numpy()
    sm = st.smooth_returns(r, theta=0.5)
    assert abs(sm.mean() - r.mean()) / abs(r.mean()) < 0.05  # mean ~ preserved


def test_smoothing_cuts_measured_vol_and_adds_autocorr(null_tape):
    r = null_tape.to_numpy()
    sm = st.smooth_returns(r, theta=0.5)
    assert sm.std(ddof=1) < r.std(ddof=1)          # measured vol falls
    ac = st._autocorr(sm, 1)
    assert ac > 0.3                                 # positive autocorrelation injected


def test_smoothing_inflates_naive_not_honest(null_tape):
    r = null_tape.to_numpy()
    sm = st.smooth_returns(r, theta=0.5)
    naive_gain = st.sharpe_naive(sm) - st.sharpe_naive(r)
    honest_gain = st.sharpe_honest_annual(sm) - st.sharpe_honest_annual(r)
    assert naive_gain > 0.10          # the reported Sharpe balloons
    assert abs(honest_gain) < 0.05    # the honest Sharpe is immune


def test_smoothing_inflation_grows_with_theta(null_tape):
    r = null_tape.to_numpy()
    sw = st.theta_sweep(r, thetas=(0.0, 0.3, 0.6, 0.8))
    assert sw["naive_sharpe"].is_monotonic_increasing
    assert sw["honest_sharpe"].max() - sw["honest_sharpe"].min() < 0.05  # honest flat


# ---- lever 2: LEVERAGE changes NEITHER Sharpe (the null lever) -------------
def test_leverage_leaves_both_sharpes_unchanged(null_tape):
    r = null_tape.to_numpy()
    lev = st.lever_returns(r, leverage=3.0)
    assert abs(st.sharpe_naive(lev) - st.sharpe_naive(r)) < 1e-9
    assert abs(st.sharpe_honest_annual(lev) - st.sharpe_honest_annual(r)) < 1e-9


# ---- lever 3: VOL-TARGETING is no free lunch, and costs cannot help --------
def test_vol_target_no_lookahead_and_costs_dont_raise(null_tape):
    r = null_tape.to_numpy()
    c = st.net_vol_target(r, target_ann_vol=0.10, cost_bps=2.0)
    assert c["net_naive_sharpe"] <= c["gross_naive_sharpe"] + 1e-9
    assert 0.0 <= c["turnover_per_day"] <= 1.0


# ---- the bootstrap bands: honest inflation straddles 0 ---------------------
def test_bootstrap_honest_band_straddles_zero(null_tape):
    nl = st.smoothing_inflation_null(null_tape.to_numpy(), theta=0.5, n_boot=400, seed=590)
    lo, hi = nl["honest_ci"]
    assert lo < 0.0 < hi                       # honest inflation CI contains zero
    assert nl["naive_inflation"] > 0.10        # naive inflation firmly positive
    assert nl["frac_honest_ge_naive"] < 0.05   # honest never mimics the game


# ---- the control: the HONEST Sharpe tracks REAL edge ----------------------
def test_honest_sharpe_tracks_planted_alpha(null_tape, edge_tape):
    h_null = st.sharpe_honest_annual(null_tape.to_numpy())
    h_edge = st.sharpe_honest_annual(edge_tape.to_numpy())
    assert h_edge > h_null + 0.3               # planting real edge lifts the honest Sharpe


def test_seed_robust_control_honest_tracks_alpha_naive_game_persists():
    lo = st.seed_robust_control(data, alpha=0.0, n_seeds=20)
    hi = st.seed_robust_control(data, alpha=0.20, n_seeds=20)
    # honest raw Sharpe rises with real edge:
    assert hi["mean_honest_raw"] > lo["mean_honest_raw"] + 0.3
    # smoothing still inflates the NAIVE Sharpe regardless of alpha (the game persists):
    assert lo["mean_naive_inflation"] > 0.10
    assert hi["mean_naive_inflation"] > 0.10
    # but the honest inflation stays economically ~0 at both:
    assert abs(lo["mean_honest_inflation"]) < 0.05
    assert abs(hi["mean_honest_inflation"]) < 0.05
