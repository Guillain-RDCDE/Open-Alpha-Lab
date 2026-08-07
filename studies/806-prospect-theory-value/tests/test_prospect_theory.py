"""Offline, fixed-seed tests for the prospect-theory (TK) machinery.

The synthetic panel is deterministic; the Tversky-Kahneman value/weighting functions
behave (loss aversion, tail overweighting, decision weights sum to one); a lottery-like
right-skewed tape scores a higher TK value than its left-skewed mirror; the monthly sort
recovers a planted negative TK->return relation (positive long-low/short-high spread);
the null shows nothing; the sort is point-in-time (truncating the future leaves earlier
months' spreads unchanged); the timer costs reduce the net; the inference primitives
behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from prospect_theory import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0020, seed=806, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_null_at_knob_zero_still_has_dispersion(null_world):
    # The TK value must vary across names even in the null (the sort has something to bite on),
    # but predict nothing (checked in test_null_world_no_signal).
    ret = st.close_returns(null_world)
    R = ret.to_numpy(dtype=float)
    tk = np.array([st.tk_value(R[-252:, j]) for j in range(R.shape[1])])
    assert np.nanstd(tk) > 0


# --------------------------------------------------------------------------- #
# The Tversky-Kahneman value & weighting primitives
# --------------------------------------------------------------------------- #
def test_loss_aversion_in_value_function():
    # v(-x) should be 2.25x steeper than v(x): a single -x outcome vs a single +x outcome.
    x = np.array([0.05])
    gain = st.tk_value(x, min_obs=1)
    loss = st.tk_value(-x, min_obs=1)
    assert loss < 0 < gain
    assert abs(loss) > 2.0 * abs(gain)          # loss aversion (~2.25)


def test_decision_weights_telescope_and_are_nonnegative():
    # The Tversky-Kahneman decision weights are rank-dependent: they are non-negative and
    # telescope to the weighted tail mass (gain side -> w+(P_gain), loss side -> w-(P_loss)).
    # They are deliberately SUBADDITIVE (do not sum to one — the CPT subcertainty feature).
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, 400)
    xs = np.sort(x)
    n = len(xs)
    i = np.arange(1, n + 1)
    loss_w = st._weight(i / n, st.TK_DELTA) - st._weight((i - 1) / n, st.TK_DELTA)
    j = n - i + 1
    gain_w = st._weight(j / n, st.TK_GAMMA) - st._weight((j - 1) / n, st.TK_GAMMA)
    is_gain = xs >= 0
    n_gain = int(is_gain.sum())
    assert (loss_w[~is_gain] >= -1e-12).all() and (gain_w[is_gain] >= -1e-12).all()
    assert abs(gain_w[is_gain].sum() - st._weight(n_gain / n, st.TK_GAMMA)) < 1e-9
    assert abs(loss_w[~is_gain].sum() - st._weight((n - n_gain) / n, st.TK_DELTA)) < 1e-9
    total = np.where(is_gain, gain_w, loss_w).sum()
    assert 0.0 < total <= 1.0 + 1e-9        # subadditive: strictly below one here


def test_lottery_scores_higher_than_its_mirror():
    # A right-skewed lottery tape (rare big gains) must score a HIGHER TK value than its
    # left-skewed mirror (rare big losses).
    rng = np.random.default_rng(2)
    lottery = np.concatenate([rng.normal(-0.004, 0.008, 480), rng.normal(0.12, 0.02, 20)])
    crash = -lottery
    assert st.tk_value(lottery) > st.tk_value(crash)


# --------------------------------------------------------------------------- #
# The cross-sectional sort
# --------------------------------------------------------------------------- #
def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.tk_stats(st.tk_spreads(ret, win_days=252))
    assert ts["t_nw"] > 2.0             # long-low-TK / short-high-TK spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-TK names out-earn high-TK names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.tk_stats(st.tk_spreads(ret, win_days=252))
    assert abs(ts["t_nw"]) < 2.5


def test_full_window_required():
    # No spread month may be formed before a full trailing window is available.
    ret = st.close_returns(data.synthetic_panel(edge=0.0, seed=806, n_assets=10, n_days=700))
    sp = st.tk_spreads(ret, win_days=252)
    first = sp.index.min()
    # the first sorted month-end must sit at least win_days trading days into the sample
    assert (ret.index < first).sum() >= 252


def test_sort_is_point_in_time():
    # Truncating the FUTURE must not change earlier months' spreads (no look-ahead beyond
    # the intended one-month forward return).
    full = data.synthetic_panel(edge=0.0020, seed=806, n_assets=20, n_days=900)
    cut = {k: v.iloc[:820].copy() for k, v in full.items()}
    spF = st.tk_spreads(st.close_returns(full), win_days=252)
    spC = st.tk_spreads(st.close_returns(cut), win_days=252)
    # drop the truncation-boundary month of the cut series (its forward month is a partial,
    # data-truncated month); every earlier month must match the full-sample spread exactly.
    common = spC.index.intersection(spF.index).sort_values()[:-1]
    assert len(common) > 5
    assert np.allclose(spF.loc[common, "spread"].to_numpy(),
                       spC.loc[common, "spread"].to_numpy())


def test_last_month_without_forward_is_dropped(edge_world):
    ret = st.close_returns(edge_world)
    me = st._month_end_positions(ret.index)
    last_me = ret.index[me[-1]]
    sp = st.tk_spreads(ret, win_days=252)
    assert sp.index.max() < last_me      # the no-forward final month is excluded


# --------------------------------------------------------------------------- #
# The timer
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.tk_spreads(ret, win_days=252)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_direction():
    a = np.array([2.0, 3.0, 2.5, 3.5, 2.2])
    b = np.array([0.0, 1.0, 0.5, -0.5, 0.2])
    assert st.welch_t(a, b) > 0
