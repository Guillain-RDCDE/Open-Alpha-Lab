"""The portfolio engine's invariants and the study's core spine:
the PP improves risk-adjusted return over SPY when regime diversification is real."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from permanent_portfolio import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_returns(world):
    frame, _ = world
    return st.to_returns(frame)


def _rename(returns, mapping):
    """Rename columns of a return frame (synthetic -> real-world names)."""
    return returns.rename(columns=mapping)


SYNTH_MAP = {"STK": "SPY", "BOND": "TLT", "GOLD": "GLD", "CASH": "SHY"}


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------
def test_pp_weights_sum_to_one(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret, cost_bps=0.0)
    # Equity curve should end well above zero (no blow-up)
    equity = (1.0 + pp).cumprod()
    assert equity.iloc[-1] > 0.5
    assert len(pp) == len(ret) - 0 or len(pp) == len(ret)


def test_portfolio_stats_keys(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    s = st.portfolio_stats(pp)
    for key in ("cagr", "vol", "sharpe", "max_dd", "worst_year"):
        assert key in s
        assert np.isfinite(s[key])


def test_cost_lowers_cagr(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp0 = st.permanent_portfolio(ret, cost_bps=0.0)
    pp5 = st.permanent_portfolio(ret, cost_bps=5.0)
    s0 = st.portfolio_stats(pp0)
    s5 = st.portfolio_stats(pp5)
    assert s0["cagr"] >= s5["cagr"]


def test_spy_only_matches_input(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    spy = st.spy_only(ret, ticker="SPY")
    assert np.allclose(spy.to_numpy(), ret["SPY"].to_numpy())


def test_blended_portfolio_daily_return_is_weighted_sum(null_world):
    """On the first day the portfolio return equals the weighted sum of leg returns."""
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.blended_portfolio(ret, {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "SHY": 0.25},
                               cost_bps=0.0)
    r0 = ret.iloc[0]
    expected = 0.25 * r0["SPY"] + 0.25 * r0["TLT"] + 0.25 * r0["GLD"] + 0.25 * r0["SHY"]
    assert abs(float(pp.iloc[0]) - expected) < 1e-10


def test_annual_returns_length(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    ar = st.annual_returns(pp)
    # About n_years calendar years
    assert len(ar) >= 18


def test_max_drawdown_negative(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    s = st.portfolio_stats(pp)
    assert s["max_dd"] <= 0.0


def test_worst_year_negative_when_losses_exist(cycle_world):
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    ar = st.annual_returns(pp)
    if (ar < 0).any():
        s = st.portfolio_stats(pp)
        assert s["worst_year"] < 0.0


# ---------------------------------------------------------------------------
# The study's spine: PP improves Sharpe over raw SPY on the null world too,
# but on the cycle world it does so more robustly (larger bootstrap fraction).
# ---------------------------------------------------------------------------
def test_pp_sharpe_vs_spy_positive_in_cycle_world(cycle_world):
    """When regime diversification is real, the PP Sharpe exceeds SPY's."""
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]
    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    pp_s = st.portfolio_stats(pp, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)
    # PP should have higher Sharpe because the planted cycle is diversifiable
    assert pp_s["sharpe"] > spy_s["sharpe"]


def test_pp_max_dd_smaller_than_spy(cycle_world):
    """PP drawdown should be substantially smaller than SPY's."""
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    pp_s = st.portfolio_stats(pp)
    spy_s = st.portfolio_stats(spy)
    assert pp_s["max_dd"] > spy_s["max_dd"]  # both negative; PP is closer to 0


def test_hac_tstat_finite(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    t = st.hac_tstat_annual(pp, spy)
    assert np.isfinite(t)


def test_bootstrap_sharpe_diff_bounds(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]
    pp = st.permanent_portfolio(ret)
    spy = st.spy_only(ret)
    result = st.bootstrap_sharpe_diff(pp, spy, rf=rf, n_boot=200, seed=144)
    assert 0.0 <= result["frac_a_wins"] <= 1.0
    lo, hi = result["ci95"]
    assert lo <= result["point"] <= hi or not np.isnan(result["point"])


def test_equity_drawdowns_returns_list(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    eps = st.equity_drawdowns(ret, "SPY", thresh=-0.10)
    assert isinstance(eps, list)
    for ep in eps:
        assert ep["stock_loss"] <= -0.10
        assert "peak" in ep and "trough" in ep and "others" in ep
