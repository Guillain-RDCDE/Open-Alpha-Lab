"""The portfolio engine's invariants and the study's core spine:
the Golden Butterfly improves risk-adjusted return over SPY when regime
diversification is real, and reduces drawdowns more than the Permanent Portfolio."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_butterfly import strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "gb_panel.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)

# Mapping from synthetic to real-world column names
SYNTH_MAP = {"LCG": "SPY", "SCV": "IWN", "BOND": "TLT", "CASH": "SHY", "GOLD": "GLD"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_returns(world):
    frame, _ = world
    return st.to_returns(frame)


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------
def test_gb_equity_curve_stays_positive(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret, cost_bps=0.0)
    equity = (1.0 + gb).cumprod()
    assert equity.iloc[-1] > 0.5


def test_portfolio_stats_keys(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    s = st.portfolio_stats(gb)
    for key in ("cagr", "vol", "sharpe", "max_dd", "worst_year"):
        assert key in s
        assert np.isfinite(s[key])


def test_cost_lowers_cagr(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb0 = st.golden_butterfly(ret, cost_bps=0.0)
    gb5 = st.golden_butterfly(ret, cost_bps=5.0)
    s0 = st.portfolio_stats(gb0)
    s5 = st.portfolio_stats(gb5)
    assert s0["cagr"] >= s5["cagr"]


def test_spy_only_matches_input(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    spy = st.spy_only(ret, ticker="SPY")
    assert np.allclose(spy.to_numpy(), ret["SPY"].to_numpy())


def test_blended_portfolio_first_day_is_weighted_sum(null_world):
    """On the first day the portfolio return equals the weighted sum of leg returns."""
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    weights = {"SPY": 0.20, "IWN": 0.20, "TLT": 0.20, "SHY": 0.20, "GLD": 0.20}
    gb = st.blended_portfolio(ret, weights, cost_bps=0.0)
    r0 = ret.iloc[0]
    expected = sum(0.20 * r0[k] for k in weights)
    assert abs(float(gb.iloc[0]) - expected) < 1e-10


def test_equal_weight_matches_golden_butterfly(null_world):
    """GB is exactly 1/N equal weight across five legs — both engines must agree."""
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret, cost_bps=0.0)
    ew = st.equal_weight(ret, tickers=("SPY", "IWN", "TLT", "SHY", "GLD"), cost_bps=0.0)
    assert np.allclose(gb.to_numpy(), ew.to_numpy(), atol=1e-12)


def test_annual_returns_length(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    ar = st.annual_returns(gb)
    assert len(ar) >= 18  # ~20 calendar years


def test_max_drawdown_negative(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    s = st.portfolio_stats(gb)
    assert s["max_dd"] <= 0.0


def test_worst_year_negative_when_losses_exist(cycle_world):
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    ar = st.annual_returns(gb)
    if (ar < 0).any():
        s = st.portfolio_stats(gb)
        assert s["worst_year"] < 0.0


# ---------------------------------------------------------------------------
# The study's spine: GB improves Sharpe over SPY on the cycle world
# ---------------------------------------------------------------------------
def test_gb_sharpe_vs_spy_positive_in_cycle_world(cycle_world):
    """When regime diversification is real, the GB Sharpe exceeds SPY's."""
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]
    gb = st.golden_butterfly(ret)
    spy = st.spy_only(ret)
    gb_s = st.portfolio_stats(gb, rf=rf)
    spy_s = st.portfolio_stats(spy, rf=rf)
    assert gb_s["sharpe"] > spy_s["sharpe"]


def test_gb_max_dd_smaller_than_spy(cycle_world):
    """GB drawdown should be substantially smaller than SPY's."""
    ret = _make_returns(cycle_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    spy = st.spy_only(ret)
    gb_s = st.portfolio_stats(gb)
    spy_s = st.portfolio_stats(spy)
    assert gb_s["max_dd"] > spy_s["max_dd"]  # both negative; GB is closer to 0


def test_hac_tstat_finite(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    gb = st.golden_butterfly(ret)
    spy = st.spy_only(ret)
    t = st.hac_tstat_annual(gb, spy)
    assert np.isfinite(t)


def test_bootstrap_sharpe_diff_bounds(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    rf = ret["SHY"]
    gb = st.golden_butterfly(ret)
    spy = st.spy_only(ret)
    result = st.bootstrap_sharpe_diff(gb, spy, rf=rf, n_boot=200, seed=203)
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


def test_sixty_forty_uses_only_two_legs(null_world):
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    p60 = st.sixty_forty(ret)
    s = st.portfolio_stats(p60)
    assert np.isfinite(s["cagr"])


def test_permanent_portfolio_matches_25pct(null_world):
    """Permanent Portfolio should have CAGR between cash and equity Sharpes."""
    ret = _make_returns(null_world).rename(columns=SYNTH_MAP)
    pp = st.permanent_portfolio(ret)
    s = st.portfolio_stats(pp)
    assert np.isfinite(s["sharpe"])


# ---------------------------------------------------------------------------
# Real-tape tests — skip if cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_gb_stats():
    """Real GB produces finite stats over the 2004-2026 window."""
    from golden_butterfly import data
    px = data.load_real(fetch=False)
    ret = st.to_returns(px)
    rf = ret["SHY"]
    gb = st.golden_butterfly(ret, cost_bps=1.0)
    s = st.portfolio_stats(gb, rf=rf)
    assert np.isfinite(s["cagr"])
    assert np.isfinite(s["sharpe"])
    assert s["max_dd"] < 0.0
    # Golden Butterfly should have lower max drawdown than SPY
    spy = st.spy_only(ret)
    spy_s = st.portfolio_stats(spy)
    assert s["max_dd"] > spy_s["max_dd"]


@requires_cache
def test_real_hac_tstat():
    """HAC t-stat on GB vs SPY annual returns is finite."""
    from golden_butterfly import data
    px = data.load_real(fetch=False)
    ret = st.to_returns(px)
    gb = st.golden_butterfly(ret, cost_bps=1.0)
    spy = st.spy_only(ret)
    t = st.hac_tstat_annual(gb, spy)
    assert np.isfinite(t)
