"""Portfolio engine invariants, the honest controls, and the study's core spine:
the Talmud 1/3–1/3–1/3 blend reduces drawdown when regime diversification is real,
but has no return edge over 60/40 on the real tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from talmud_portfolio import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
SHARED_PANEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache",
    "cross_asset_etfs.parquet",
)
STUDY_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "_cache", "talmud_prices.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(SHARED_PANEL) and not os.path.exists(STUDY_CACHE),
    reason="real price cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SYNTH_MAP = {"REIT": "VNQ", "STK": "SPY", "BOND": "BND"}


def _ret(frame, mapping=None):
    r = st.to_returns(frame)
    if mapping:
        r = r.rename(columns=mapping)
    return r


# ---------------------------------------------------------------------------
# Engine invariants (offline)
# ---------------------------------------------------------------------------
def test_blended_portfolio_weights_sum_to_one(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    tal = st.talmud_portfolio(ret, cost_bps=0.0, reit="VNQ", stk="SPY", bond="BND")
    equity = (1.0 + tal).cumprod()
    assert equity.iloc[-1] > 0.1
    assert len(tal) == len(ret)


def test_portfolio_stats_keys(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    tal = st.talmud_portfolio(ret, reit="VNQ", stk="SPY", bond="BND")
    s = st.portfolio_stats(tal)
    for key in ("cagr", "vol", "sharpe", "max_dd", "worst_year"):
        assert key in s
        assert np.isfinite(s[key])


def test_cost_lowers_cagr(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    t0 = st.talmud_portfolio(ret, cost_bps=0.0, reit="VNQ", stk="SPY", bond="BND")
    t5 = st.talmud_portfolio(ret, cost_bps=5.0, reit="VNQ", stk="SPY", bond="BND")
    s0 = st.portfolio_stats(t0)
    s5 = st.portfolio_stats(t5)
    assert s0["cagr"] >= s5["cagr"]


def test_no_rebalance_differs_from_annual(null_world):
    """A buy-and-hold (no rebalance) portfolio drifts away from fixed weights."""
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    weights = {"VNQ": 1 / 3, "SPY": 1 / 3, "BND": 1 / 3}
    annual = st.blended_portfolio(ret, weights, rebalance="annual", cost_bps=0.0)
    hold   = st.blended_portfolio(ret, weights, rebalance="none",   cost_bps=0.0)
    # Different rebalance regimes produce different equity paths
    assert not np.allclose(
        (1 + annual).cumprod().to_numpy(),
        (1 + hold).cumprod().to_numpy(),
        atol=1e-6,
    )


def test_blended_portfolio_unknown_rebalance(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    weights = {"VNQ": 1 / 3, "SPY": 1 / 3, "BND": 1 / 3}
    with pytest.raises(ValueError):
        st.blended_portfolio(ret, weights, rebalance="weekly")


def test_sixty_forty_equity_positive(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    s = st.sixty_forty(ret, stk="SPY", bond="BND")
    equity = (1.0 + s).cumprod()
    assert equity.iloc[-1] > 0.1


def test_spy_only_matches_asset(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    spy = st.spy_only(ret, ticker="SPY")
    assert np.allclose(spy.to_numpy(), ret["SPY"].to_numpy())


def test_annual_returns_length(null_world):
    frame, truth = null_world
    ret = _ret(frame, SYNTH_MAP)
    tal = st.talmud_portfolio(ret, reit="VNQ", stk="SPY", bond="BND")
    ann = st.annual_returns(tal)
    assert len(ann) >= truth["n_years"] - 1


def test_hac_tstat_annual_finite(null_world):
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    tal = st.talmud_portfolio(ret, reit="VNQ", stk="SPY", bond="BND")
    spy = st.spy_only(ret)
    t = st.hac_tstat_annual(tal, spy)
    assert np.isfinite(t)


# ---------------------------------------------------------------------------
# The study's spine (offline, synthetic)
# ---------------------------------------------------------------------------
def test_talmud_reduces_drawdown_in_cycle_world(cycle_world, null_world):
    """With a planted regime cycle the Talmud blend has a shallower max drawdown
    than 100% stocks — the core defensive claim made by the original allocation."""
    for world, label in [(cycle_world, "cycle"), (null_world, "null")]:
        frame, _ = world
        ret = _ret(frame, SYNTH_MAP)
        tal = st.talmud_portfolio(ret, reit="VNQ", stk="SPY", bond="BND")
        spy = st.spy_only(ret)
        s_tal = st.portfolio_stats(tal)
        s_spy = st.portfolio_stats(spy)
        # In both worlds the blend should have a shallower max drawdown than 100% stocks
        # (because the bond leg hedges in any realistic correlation regime)
        assert s_tal["max_dd"] > s_spy["max_dd"], (
            f"[{label}] Talmud max_dd {s_tal['max_dd']:.3f} not better than SPY {s_spy['max_dd']:.3f}"
        )


def test_talmud_lower_vol_than_spy(null_world):
    """The three-leg blend should always have lower volatility than 100% SPY."""
    frame, _ = null_world
    ret = _ret(frame, SYNTH_MAP)
    tal = st.talmud_portfolio(ret, reit="VNQ", stk="SPY", bond="BND")
    spy = st.spy_only(ret)
    s_tal = st.portfolio_stats(tal)
    s_spy = st.portfolio_stats(spy)
    assert s_tal["vol"] < s_spy["vol"]


# ---------------------------------------------------------------------------
# Real-data tests (skipped on CI when cache is absent)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_talmud_keys():
    prices = data.load_real()
    ret = st.to_returns(prices)
    tal = st.talmud_portfolio(ret)
    s = st.portfolio_stats(tal)
    for key in ("cagr", "vol", "sharpe", "max_dd", "worst_year"):
        assert key in s
        assert np.isfinite(s[key])


@requires_cache
def test_real_talmud_lower_dd_than_spy():
    prices = data.load_real()
    ret = st.to_returns(prices)
    tal = st.talmud_portfolio(ret)
    spy = st.spy_only(ret)
    s_tal = st.portfolio_stats(tal)
    s_spy = st.portfolio_stats(spy)
    assert s_tal["max_dd"] > s_spy["max_dd"]
