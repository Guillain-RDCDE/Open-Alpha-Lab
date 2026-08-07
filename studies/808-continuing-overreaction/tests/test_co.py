"""Offline, fixed-seed tests for the continuing-overreaction machinery.

The synthetic panel is deterministic; the CO score is a normalised weighted count of
monthly-return signs (bounded, recency-weighted); the sort recovers a planted
continuation (positive long-high-CO / short-low-CO spread); the null shows nothing; the
sort is point-in-time (skip-month lag, no look-ahead); the timer costs reduce the net;
the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from continuing_overreaction import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.02, seed=808, n_assets=40, n_days=1800)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_co_is_bounded_and_weighted():
    # An all-up name scores +1, an all-down name scores -1 (weights sum to 1).
    idx = pd.bdate_range("2010-01-01", periods=400)
    up = pd.DataFrame({"A": np.linspace(100, 300, 400)}, index=idx)
    dn = pd.DataFrame({"A": np.linspace(300, 100, 400)}, index=idx)
    m_up = st.monthly_returns({"A": up.rename(columns={"A": "Close"})})
    m_dn = st.monthly_returns({"A": dn.rename(columns={"A": "Close"})})
    co_up = st.co_signal(m_up).to_numpy()
    co_dn = st.co_signal(m_dn).to_numpy()
    co_up = co_up[~np.isnan(co_up)]
    co_dn = co_dn[~np.isnan(co_dn)]
    assert np.allclose(co_up, 1.0)
    assert np.allclose(co_dn, -1.0)
    # Every score lives in [-1, 1].
    assert (np.abs(co_up) <= 1.0 + 1e-9).all()


def test_recent_month_weighted_more():
    # Flip only the most-recent in-window month's sign vs only the oldest; the recent
    # flip must move CO more (weights increase toward the recent months).
    n, skip = 12, 1
    offset = n + skip                     # last row i = offset uses window rows 0..offset-2
    m_base = pd.DataFrame(0.01 * np.ones((offset + 1, 1)))   # i = offset (last row)
    co_base = st.co_signal(m_base, n, skip).to_numpy()[-1, 0]
    oldest_row = 0                        # window row i-offset  (oldest, smallest weight)
    recent_row = offset - 2               # window row i-2       (most recent, largest weight)
    m_old = m_base.copy(); m_old.iloc[oldest_row] = -0.01
    co_old = st.co_signal(m_old, n, skip).to_numpy()[-1, 0]
    m_rec = m_base.copy(); m_rec.iloc[recent_row] = -0.01
    co_rec = st.co_signal(m_rec, n, skip).to_numpy()[-1, 0]
    assert (co_base - co_rec) > (co_base - co_old) > 0


def test_planted_relation_recovered(edge_world):
    monthly = st.monthly_returns(edge_world)
    ts = st.co_stats(st.co_spreads(monthly))
    assert ts["t_nw"] > 3.0             # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-CO names out-earn low-CO names


def test_null_world_no_signal(null_world):
    monthly = st.monthly_returns(null_world)
    ts = st.co_stats(st.co_spreads(monthly))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    # co_signal on row i must depend only on returns through row i-1-skip.
    rng = np.random.default_rng(0)
    monthly = pd.DataFrame(
        rng.normal(0, 0.05, (40, 3)),
        index=pd.period_range("2015-01", periods=40, freq="M").to_timestamp("M"),
        columns=["A", "B", "C"],
    )
    n, skip = 12, 1
    co = st.co_signal(monthly, n, skip)
    # Perturb the most-recent held-month row far in the future; earlier CO unchanged.
    perturbed = monthly.copy(); perturbed.iloc[30:] *= -3.0
    co2 = st.co_signal(perturbed, n, skip)
    # rows that only use returns before row 30 must be identical
    upto = 30 - (n + skip)
    assert np.allclose(co.iloc[:upto].to_numpy(), co2.iloc[:upto].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    monthly = st.monthly_returns(edge_world)
    sp = st.co_spreads(monthly)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_sign_of_monthly_returns_used_not_magnitude():
    # CO reads only signs: rescaling a name's positive returns leaves its CO unchanged.
    idx = pd.period_range("2015-01", periods=30, freq="M").to_timestamp("M")
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.normal(0.01, 0.04, (30, 1)), index=idx)
    scaled = base.copy()
    pos = scaled.to_numpy() > 0
    scaled.iloc[:, 0] = np.where(pos[:, 0], scaled.iloc[:, 0] * 5.0, scaled.iloc[:, 0])
    co_b = st.co_signal(base).to_numpy()
    co_s = st.co_signal(scaled).to_numpy()
    assert np.allclose(co_b, co_s, equal_nan=True)


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi
