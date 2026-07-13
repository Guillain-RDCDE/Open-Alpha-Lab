"""Offline, deterministic tests for Study 766 — Memecoin-Season.

Pure-function checks on synthetic/constructed data — NO network, no cache dependency. Covers
the honesty rails (cost monotonicity, zero-cost == gross, one-week lag / no look-ahead) and the
study's spine (the momentum rotation harvests planted persistence and reads ~0 on a null world).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from memecoin_season import data, strategy as st  # noqa: E402


def _synthetic_prices(n=120, seed=1):
    """A deterministic BTC/DOGE/SHIB-shaped weekly price frame (no network)."""
    rets = data.synthetic_world(n_weeks=n, persistence=0.0, seed=seed)
    rets.columns = data.ASSETS
    px = (1.0 + rets).cumprod() * 100.0
    px.index = pd.date_range("2021-01-01", periods=n, freq="W-FRI")
    return px


# --------------------------------------------------------------------------- #
# Signal construction & the one documented lag
# --------------------------------------------------------------------------- #
def test_momentum_signal_matches_pct_change():
    px = _synthetic_prices()
    mom = st.momentum_signal(px, lookback=4)
    expected = px["BTC"].iloc[4] / px["BTC"].iloc[0] - 1.0
    assert abs(mom["BTC"].iloc[4] - expected) < 1e-12


def test_rotation_choice_is_argmax():
    px = _synthetic_prices()
    mom = st.momentum_signal(px, lookback=4).dropna(how="all")
    choice = st.rotation_choice(px, lookback=4)
    t = choice.index[10]
    assert choice.loc[t] == mom.loc[t].idxmax()


def test_execution_lag_no_lookahead():
    """The coin held during week t must be the one chosen at t-1 (choice.shift(1))."""
    px = _synthetic_prices()
    choice = st.rotation_choice(px, lookback=4)
    res = st.run_rotation(px, lookback=4, cost_bps=0.0)
    held = res["held"]
    # held[t] equals choice[t_prev] where t_prev is the previous decision week
    common = held.index[5]
    prev = choice.index[choice.index.get_loc(common) - 1]
    assert held.loc[common] == choice.loc[prev]


# --------------------------------------------------------------------------- #
# Cost honesty rails
# --------------------------------------------------------------------------- #
def test_zero_cost_net_equals_gross():
    px = _synthetic_prices()
    res = st.run_rotation(px, lookback=4, cost_bps=0.0)
    np.testing.assert_allclose(res["net_ret"].values, res["gross_ret"].values, atol=1e-12)


def test_costs_reduce_returns():
    px = _synthetic_prices()
    res = st.run_rotation(px, lookback=4, cost_bps=50.0)
    assert (res["net_ret"] <= res["gross_ret"] + 1e-12).all()
    assert (res["cost"] >= -1e-12).all()


def test_more_cost_is_weakly_worse():
    px = _synthetic_prices()
    lo = st.summarize(st.run_rotation(px, cost_bps=10.0)["net_ret"])["total_pct"]
    hi = st.summarize(st.run_rotation(px, cost_bps=100.0)["net_ret"])["total_pct"]
    assert hi <= lo + 1e-9


def test_turnover_bounds():
    px = _synthetic_prices()
    res = st.run_rotation(px, lookback=4, cost_bps=30.0)
    assert res["turnover"].max() <= 2.0 + 1e-9
    assert res["turnover"].min() >= -1e-9


# --------------------------------------------------------------------------- #
# Summary + inference primitives
# --------------------------------------------------------------------------- #
def test_summarize_keys():
    px = _synthetic_prices()
    s = st.summarize(st.btc_hodl(px))
    for k in ["n", "total_pct", "cagr_pct", "vol_pct", "sharpe", "maxdd_pct", "hit_rate"]:
        assert k in s


def test_excess_tstat_zero_against_self():
    """Excess of a series against itself has zero spread -> undefined t (guarded to NaN)."""
    px = _synthetic_prices()
    btc = st.btc_hodl(px)
    ex = st.excess_tstat(btc, btc)
    # zero-variance difference: the guard returns NaN rather than dividing by zero
    assert np.isnan(ex["t"])
    # and a non-degenerate shifted comparison has a finite, small-ish mean excess
    ex2 = st.excess_tstat(btc * 1.0, st.equal_weight(px))
    assert np.isfinite(ex2["t"])


# --------------------------------------------------------------------------- #
# The spine — control detects planted momentum, reads ~0 on the null
# --------------------------------------------------------------------------- #
def test_control_null_near_zero():
    ts = np.array([st.momentum_edge_from_returns(
        data.synthetic_world(persistence=0.0, seed=766 + s)) for s in range(20)])
    assert abs(ts.mean()) < 1.0, f"null mean t too large: {ts.mean():.2f}"


def test_control_recovers_planted_persistence():
    t = st.momentum_edge_from_returns(data.synthetic_world(persistence=0.35, seed=766))
    assert t > 2.0, f"planted-persistence rotation should beat equal-weight (t>2), got {t:.2f}"


def test_synthetic_world_deterministic():
    a = data.synthetic_world(persistence=0.2, seed=5)
    b = data.synthetic_world(persistence=0.2, seed=5)
    pd.testing.assert_frame_equal(a, b)
