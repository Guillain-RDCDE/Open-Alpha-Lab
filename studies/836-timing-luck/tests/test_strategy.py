"""Offline, fixed-seed tests for the timing-luck machinery.

The pitfall appears at its planted magnitude — the SAME momentum book traced on
different rebalance offsets shows a **material Sharpe dispersion** (a phantom gap), the
lucky offset does **not** persist out-of-sample (pure luck), and **tranching collapses
the dispersion** to a single curve. The tranched book is silent on the null and lights
up on a planted momentum premium; costs reduce the net; the inference primitives behave.
All offline.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from timing_luck import data, strategy as st  # noqa: E402


# ---- signal + weights ------------------------------------------------------
def test_ls_weights_dollar_neutral():
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    w = st._ls_weights(sig, top_frac=0.3, min_names=6)
    assert abs(w.sum()) < 1e-12          # net zero (dollar neutral)
    assert abs(w[w > 0].sum() - 1.0) < 1e-12   # long leg sums to +1
    assert abs(w[w < 0].sum() + 1.0) < 1e-12   # short leg sums to -1
    assert w[9] > 0 and w[0] < 0         # highest momentum long, lowest short


def test_ls_weights_too_few_names_all_zero():
    sig = np.array([1.0, 2.0, np.nan, np.nan])
    w = st._ls_weights(sig, top_frac=0.3, min_names=6)
    assert np.all(w == 0.0)


def test_trailing_return_is_point_in_time():
    R = np.tile(np.linspace(-0.01, 0.01, 5), (40, 1))  # (40 days, 5 names)
    tr = st.trailing_return(R, lookback=10)
    assert np.all(np.isnan(tr[:10]))     # no signal before the window fills
    # row t equals the log-return sum over the trailing window
    lp = np.cumsum(np.log1p(R), axis=0)
    assert np.allclose(tr[20], lp[20] - lp[10])


# ---- THE PITFALL: material Sharpe dispersion across offsets -----------------
def test_timing_luck_dispersion_is_material(null_world):
    tl = st.timing_luck(null_world)
    assert tl["period"] == 21
    assert len(tl["sharpes"]) == 21
    # the SAME strategy on different rebalance days prints a materially different Sharpe
    assert tl["sharpe_spread"] > 0.25
    assert tl["best_offset"] != tl["worst_offset"]


def test_dispersion_present_in_both_worlds(null_world, edge_world):
    """The phantom dispersion is an artefact of WHEN you rebalance — it is present
    whether or not there is a genuine edge underneath."""
    d_null = st.timing_luck(null_world)["sharpe_spread"]
    d_edge = st.timing_luck(edge_world)["sharpe_spread"]
    assert d_null > 0.25 and d_edge > 0.25


# ---- LUCK, NOT SKILL: the lucky offset does not persist ---------------------
def test_lucky_offset_does_not_persist(null_world):
    pr = st.offset_persistence(null_world)
    assert len(pr["sharpe_h1"]) == 21
    # first-half vs second-half offset ranking is ~uncorrelated: the winner is a coin-flip
    assert abs(pr["rank_corr"]) < 0.5


# ---- THE FIX: tranching collapses the dispersion ---------------------------
def test_tranching_collapses_dispersion(null_world):
    tl = st.timing_luck(null_world)
    tr = st.tranched_portfolio(null_world)
    # there is exactly ONE tranched curve, and it sits inside the offset spread:
    assert tl["sharpe_worst"] - 1e-9 <= tr["sharpe"] <= tl["sharpe_best"] + 1e-9
    # the pre-tranch dispersion was material; the tranched book has none (single curve)
    assert tl["sharpe_spread"] > 0.25


def test_null_tranched_is_silent(null_world):
    tr = st.tranched_portfolio(null_world)
    assert abs(tr["t_nw"]) < 2.0         # no real edge to detect on the null


def test_control_tranched_fires(edge_world):
    tr = st.tranched_portfolio(edge_world)
    assert tr["t_nw"] > 2.0              # the planted momentum premium lights up
    assert tr["sharpe"] > 0.5


# ---- costs -----------------------------------------------------------------
def test_costs_reduce_net(edge_world):
    gross = st.timer_stats(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(edge_world, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


# ---- seed-robust control: null silent, planted fires -----------------------
def test_seed_robust_null_silent_control_fires():
    lo = st.seed_robust(data, mom_edge=0.0, n_seeds=8)
    hi = st.seed_robust(data, mom_edge=1.0, n_seeds=8)
    # dispersion present in both (the artefact is independent of real edge):
    assert lo["mean_sharpe_spread"] > 0.25
    assert hi["mean_sharpe_spread"] > 0.25
    # the lucky offset never persists:
    assert abs(lo["mean_rank_corr"]) < 0.4
    # tranched: silent on the null, robustly positive with a planted premium:
    assert lo["tranched_t_fires"] == 0
    assert hi["mean_tranched_sharpe"] > lo["mean_tranched_sharpe"] + 0.5
    assert hi["tranched_t_fires"] >= 6


# ---- inference primitives --------------------------------------------------
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.full(100, 0.02) + 1e-6
    b = np.zeros(100) + 1e-6
    assert st.welch_t(a, b) > 0


def test_sharpe_zero_on_degenerate():
    assert np.isnan(st.sharpe(np.zeros(2)))     # too few points
    assert np.isnan(st.sharpe(np.zeros(100)))   # zero variance
