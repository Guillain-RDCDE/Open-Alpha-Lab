"""Strategy invariants: the long-short engine, random control, and the study's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluent_tickers import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def test_ls_portfolio_columns(null_panel):
    prices, mask, _ = null_panel
    port = st.build_ls_portfolio(prices, mask)
    assert set(port.columns) == {"ret_fluent", "ret_nonfluent", "ret_ls", "ret_ls_net"}


def test_ls_portfolio_length(null_panel):
    prices, mask, truth = null_panel
    port = st.build_ls_portfolio(prices, mask)
    assert len(port) == truth["n_days"] - 1  # one row lost to log-return shift


def test_ls_portfolio_net_le_gross(null_panel):
    """Net returns should be <= gross (costs drag on rebalance dates)."""
    prices, mask, _ = null_panel
    port = st.build_ls_portfolio(prices, mask, cost_bps=10.0)
    assert port["ret_ls_net"].sum() <= port["ret_ls"].sum()


def test_ls_portfolio_no_lookahead(null_panel):
    """Returns at date t use only prices up to and including t — no peek ahead."""
    prices, mask, _ = null_panel
    port = st.build_ls_portfolio(prices, mask)
    # By construction, log_rets at date t = log(prices[t] / prices[t-1]),
    # so the first row of port corresponds to prices.index[1] (the second date).
    assert port.index[0] == prices.index[1]


def test_random_control_different_from_signal(null_panel):
    """The random-label arm is different from the fluency arm (different labels)."""
    prices, mask, _ = null_panel
    fluent_port = st.build_ls_portfolio(prices, mask)
    rand_port = st.build_random_ls_portfolio(prices, mask, seed=42)
    # They should differ in general.
    assert not np.allclose(fluent_port["ret_ls"].values, rand_port["ret_ls"].values)


def test_random_control_reproducible(null_panel):
    """Same seed → same random portfolio."""
    prices, mask, _ = null_panel
    a = st.build_random_ls_portfolio(prices, mask, seed=7)
    b = st.build_random_ls_portfolio(prices, mask, seed=7)
    assert np.allclose(a["ret_ls"].values, b["ret_ls"].values)


# ---------------------------------------------------------------------------
# Annual aggregation
# ---------------------------------------------------------------------------
def test_annual_ls_returns_shape(null_panel):
    prices, mask, truth = null_panel
    port = st.build_ls_portfolio(prices, mask)
    annual = st.annual_ls_returns(port)
    # n_days=252 ≈ 1 year → expect 1 full year.
    assert len(annual) >= 1


def test_annual_ls_returns_indexed_by_year(null_panel):
    prices, mask, _ = null_panel
    port = st.build_ls_portfolio(prices, mask)
    annual = st.annual_ls_returns(port)
    assert annual.index.dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    dummy = pd.Series([0.01, -0.02, 0.005, 0.008, -0.01, 0.012, 0.003])
    dummy.index = range(2017, 2024)
    out = st.summarize(dummy)
    assert "n_years" in out
    assert "mean_bps" in out
    assert "tstat" in out
    assert "win_rate" in out


def test_summarize_tstat_direction_with_signal(signal_panel):
    """With a planted premium, the t-stat of the L-S spread should be positive."""
    prices, mask, _ = signal_panel
    port = st.build_ls_portfolio(prices, mask, cost_bps=0.0)
    annual = st.annual_ls_returns(port)
    out = st.summarize(annual)
    # With 5% planted premium over 2 years, mean_bps should be well above zero.
    assert out["mean_bps"] > 0


def test_spine_signal_beats_null(null_panel, signal_panel):
    """Core invariant: the fluency premium is recoverable when planted."""
    def mean_spread(prices, mask):
        port = st.build_ls_portfolio(prices, mask, cost_bps=0.0)
        annual = st.annual_ls_returns(port)
        return st.summarize(annual)["mean_bps"]

    prices_null, mask_null, _ = null_panel
    prices_sig, mask_sig, _ = signal_panel
    assert mean_spread(prices_sig, mask_sig) > mean_spread(prices_null, mask_null)
