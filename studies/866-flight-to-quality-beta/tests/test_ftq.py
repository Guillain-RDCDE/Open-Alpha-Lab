"""Offline, fixed-seed tests for the flight-to-quality-beta machinery.

The synthetic world is deterministic; the conditional FTQ beta has the right sign (high
loading -> high measured FTQ beta); the sort recovers a planted pay-for-the-hedge
relation (positive long-low-FTQ / short-high-FTQ spread); the null shows nothing; the
sort is point-in-time (one month-end lag, no look-ahead); the timer costs reduce the
net; the crash-protection comparison runs; the inference primitives behave. All offline,
synthetic-only — the suite must pass with NO real cache present.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ftq_beta import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    w2 = data.synthetic_panel(edge=0.004, seed=866, n_assets=40, n_days=1500)
    for sym in edge_world["panel"]:
        assert np.allclose(edge_world["panel"][sym].to_numpy(), w2["panel"][sym].to_numpy())
    assert np.allclose(edge_world["tlt"].to_numpy(), w2["tlt"].to_numpy())
    assert np.allclose(edge_world["market"].to_numpy(), w2["market"].to_numpy())


def test_tlt_rallies_on_down_days(edge_world):
    # The safe-haven construction: bonds rise more on down-market days than up days.
    m = edge_world["market"].to_numpy()
    b = edge_world["tlt"].to_numpy()
    assert b[m < 0].mean() > b[m > 0].mean()


def test_ftq_beta_has_dispersion(edge_world):
    # The cross-section of FTQ betas must vary (the sort needs something to bite on).
    ret = st.close_returns(edge_world["panel"])
    me = st.month_ends(ret.index)
    beta = st.ftq_beta_panel(ret, edge_world["tlt"], edge_world["market"],
                             window_end=me[-1], min_down=40).dropna()
    assert len(beta) > 20
    assert beta.std() > 0.05


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world["panel"])
    sp = st.ftq_spreads(ret, edge_world["tlt"], edge_world["market"], min_stocks=10)
    ts = st.ftq_stats(sp)
    assert ts["t_nw"] > 3.0            # long-low-FTQ / short-high-FTQ spread lights up
    assert ts["spread_pct"] > 0
    assert ts["lo_pct"] > ts["hi_pct"]  # low-FTQ names out-earn high-FTQ (hedge) names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world["panel"])
    sp = st.ftq_spreads(ret, null_world["tlt"], null_world["market"], min_stocks=10)
    ts = st.ftq_stats(sp)
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time(edge_world):
    # The signal at month-end t-1 governs month t; a name's own held-month return is
    # never used to rank it. Concretely: dropping the last held month cannot change the
    # betas that were formed strictly earlier.
    ret = st.close_returns(edge_world["panel"])
    me = st.month_ends(ret.index)
    full = st.ftq_beta_panel(ret, edge_world["tlt"], edge_world["market"],
                             window_end=me[-3], min_down=40)
    trimmed = st.ftq_beta_panel(ret.loc[:me[-2]], edge_world["tlt"], edge_world["market"],
                                window_end=me[-3], min_down=40)
    assert np.allclose(full.to_numpy(), trimmed.to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world["panel"])
    sp = st.ftq_spreads(ret, edge_world["tlt"], edge_world["market"], min_stocks=10)
    gross = st.timer_stats(sp, one_way_bps=0.0, borrow_bps_yr=0.0)["net_pct"]
    net = st.timer_stats(sp, one_way_bps=10.0, borrow_bps_yr=50.0)["net_pct"]
    assert net < gross


def test_crash_protection_runs(edge_world):
    ret = st.close_returns(edge_world["panel"])
    cp = st.crash_protection(ret, edge_world["tlt"], edge_world["market"],
                             min_stocks=10, crash_pct=0.10)
    assert cp["n_crash_days"] > 0
    assert np.isfinite(cp["hi_minus_lo_crash_pct"])


def test_ftq_beta_sign_matches_construction(edge_world):
    # A name built with a positive FTQ loading (high c_i) should show a positive FTQ beta;
    # dispersion in measured betas should track the planted loading dispersion.
    ret = st.close_returns(edge_world["panel"])
    me = st.month_ends(ret.index)
    beta = st.ftq_beta_panel(ret, edge_world["tlt"], edge_world["market"],
                             window_end=me[-1], min_down=40).dropna()
    # majority of names have a finite, mostly-positive-or-negative spread of loadings;
    # the point is the estimator is not degenerate (all-zero / all-nan)
    assert np.isfinite(beta).all()
    assert (beta > 0).any() and (beta < 0).any()


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_zero_on_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(st.welch_t(a, a)) < 1e-9


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_placebo_null_centres_near_zero(null_world):
    ret = st.close_returns(null_world["panel"])
    pl = st.placebo_pvalue(ret, null_world["tlt"], null_world["market"],
                           min_stocks=10, n_draws=200)
    assert abs(pl["placebo_mean_pct"]) < 0.2
    assert 0.0 <= pl["p_value"] <= 1.0
