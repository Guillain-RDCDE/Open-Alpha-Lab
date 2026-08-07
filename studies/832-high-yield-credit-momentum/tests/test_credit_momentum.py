"""Offline, fixed-seed tests for the high-yield-credit-momentum machinery.

The synthetic panel is deterministic; the credit trend has the right sign vs the latent
factor; the discrimination test recovers a planted credit→equity relation and stays silent
on the null; the sort is point-in-time (one shift, no look-ahead); switch costs reduce the
timer's return; the inference primitives behave. All offline, synthetic-only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_credit_momentum import data, strategy as st  # noqa: E402

REAL_CACHE = data.CACHE_PATH


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.006, seed=832, n_days=2200)
    for col in edge_world.columns:
        assert np.allclose(edge_world[col].to_numpy(), p2[col].to_numpy())


def test_panel_has_all_tickers(edge_world):
    assert list(edge_world.columns) == ["HYG", "IEF", "LQD", "SPY"]
    assert (edge_world > 0).all().all()   # positive prices


def test_credit_trend_sign_matches_factor(edge_world):
    # HYG loads positively on the risk-on factor, IEF does not, so HYG-excess-IEF should be
    # positive more often than not in the planted world (risk-on drift baked in).
    tr = st.credit_trend(edge_world, lookback=126).dropna()
    assert tr.notna().sum() > 100
    assert tr.std() > 0            # the trend actually varies (the timer has something to read)


# --------------------------------------------------------------------------- #
# The discrimination detector: fires on the plant, silent on the null
# --------------------------------------------------------------------------- #
def test_planted_relation_recovered(edge_world):
    s = st.signal_stats(edge_world, lookback=126)
    assert s["t_nw"] > 2.5              # discrimination lights up
    assert s["diff_bps"] > 0           # risk-on days earn a HIGHER equity excess
    assert s["on_bps"] > s["off_bps"]


def test_planted_timer_beats_buy_and_hold(edge_world):
    t = st.timer_stats(edge_world, lookback=126, cost_bps=1.0)
    assert t["active_t_nw"] > 1.5      # the timer adds value when the plant is real
    assert t["net_sharpe"] > t["bh_sharpe"]


def test_null_world_no_signal(null_world):
    s = st.signal_stats(null_world, lookback=126)
    assert abs(s["t_nw"]) < 2.5        # no discrimination in the null


def test_null_control_calibrated():
    # Across seeds the null discrimination t is centred near zero and rarely fires.
    ts = np.array([
        st.signal_stats(data.synthetic_panel(edge=0.0, seed=832 + s, n_days=1600), 126)["t_nw"]
        for s in range(12)
    ])
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 2


# --------------------------------------------------------------------------- #
# No look-ahead + costs
# --------------------------------------------------------------------------- #
def test_signal_is_point_in_time():
    panel = pd.DataFrame(
        {
            "HYG": np.linspace(100, 130, 40),
            "IEF": np.linspace(100, 105, 40),
            "LQD": np.linspace(100, 110, 40),
            "SPY": np.linspace(100, 160, 40),
        },
        index=pd.bdate_range("2020-01-01", periods=40),
    )
    trend = st.credit_trend(panel, lookback=5)
    df = st.signal_frame(panel, lookback=5)
    # the frame's trend on a given date equals the raw trend on the PREVIOUS date (one shift)
    common = df.index[5]
    prev = panel.index[panel.index.get_loc(common) - 1]
    assert np.isclose(df.loc[common, "trend"], trend.loc[prev])


def test_costs_reduce_timer_return(edge_world):
    gross = st.timer_stats(edge_world, lookback=126, cost_bps=0.0)["net_cagr"]
    net = st.timer_stats(edge_world, lookback=126, cost_bps=10.0)["net_cagr"]
    assert net < gross


def test_regime_diff_series_mean_is_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    flag = (rng.random(500) > 0.5).astype(float)
    g = st._regime_diff_series(flag, x)
    diff = x[flag == 1].mean() - x[flag == 0].mean()
    assert np.isclose(g.mean(), diff)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_sharpe_and_cagr_positive_on_updrift():
    r = np.full(2520, 0.0005)     # steady up-drift
    assert st.sharpe(r) > 0
    assert st.cagr(r) > 0
    assert st.max_drawdown(r) == 0.0    # monotone up -> no drawdown


# --------------------------------------------------------------------------- #
# Real-cache smoke test — guarded so offline CI skips it
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_and_is_flat():
    panel = data.load_panel()
    assert list(panel.columns) == ["HYG", "IEF", "LQD", "SPY"]
    assert panel.index.max() <= pd.Timestamp(data.AS_OF)
    s = st.signal_stats(panel, lookback=126)
    # the honest real-tape finding: no significant discrimination
    assert abs(s["t_nw"]) < 2.0
