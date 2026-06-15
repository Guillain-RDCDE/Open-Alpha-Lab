"""Signals, portfolio construction invariants, and the study's spine.

The concentration rule beats equal-weight only when a real planted advantage exists;
on a null cross-section it is statistically indistinguishable from the baseline.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from megacap_concentration import strategy as st  # noqa: E402
from megacap_concentration import data            # noqa: E402


# ---- portfolio construction invariants -------------------------------------------

def test_annual_portfolio_returns_shape(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    assert len(annual) == truth["n_years"]
    assert set(["topn_ret", "ew_ret", "topn_excess", "n_stocks", "n_topn"]).issubset(annual.columns)


def test_annual_portfolio_n_topn_equals_top_n(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    # n_topn should equal top_n for every year (all stocks have returns)
    assert (annual["n_topn"] == truth["top_n"]).all()


def test_excess_is_topn_minus_ew(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    assert np.allclose(annual["topn_excess"], annual["topn_ret"] - annual["ew_ret"])


def test_ew_ret_is_mean_of_all_stocks(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    for yr, grp in df.groupby("year"):
        expected_ew = grp["ret_annual"].mean()
        actual_ew   = annual.loc[yr, "ew_ret"]
        assert abs(expected_ew - actual_ew) < 1e-10, \
            f"Year {yr}: ew_ret mismatch {expected_ew:.6f} vs {actual_ew:.6f}"


# ---- summarize invariants -------------------------------------------------------

def test_summarize_full_has_required_keys(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    result = st.summarize(annual)
    assert "full" in result
    for key in ["topn_mean", "ew_mean", "excess_mean", "excess_tstat", "n"]:
        assert key in result["full"], f"Missing key {key!r} in full stats"


def test_summarize_decade_keys(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    result = st.summarize(annual)
    if "by_decade" in result:
        assert "pre_2015" in result["by_decade"]
        assert "post_2020" in result["by_decade"]


# ---- the spine: effect knob decides whether concentration wins -------------------

def test_concentration_only_wins_with_planted_effect():
    """The spine: concentration beats equal-weight only when there's an actual reason.

    On a null cross-section (effect=0) the excess should be near zero.
    On a strongly planted cross-section (effect=0.15) the excess must be clearly positive.
    """
    # --- null: no planted advantage ------------------------------------------------
    df_null, t = data.synthetic_annual(
        n_years=50, n_stocks=200, top_n=10, effect=0.0, seed=177
    )
    df_null = df_null.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual_null = st.annual_portfolio_returns(df_null, top_n=10, col="ret_annual")
    excess_null = annual_null["topn_excess"].mean()
    # With no planted alpha, excess should be within noise of 0
    assert abs(excess_null) < 0.05, \
        f"Null excess should be near 0, got {excess_null:.4f}"

    # --- signal: strong planted effect ---------------------------------------------
    df_sig, _ = data.synthetic_annual(
        n_years=50, n_stocks=200, top_n=10, effect=0.15, seed=177
    )
    df_sig = df_sig.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual_sig = st.annual_portfolio_returns(df_sig, top_n=10, col="ret_annual")
    excess_sig = annual_sig["topn_excess"].mean()
    assert excess_sig > 0.08, \
        f"Signal excess should be clearly positive, got {excess_sig:.4f}"


def test_spy_annual_returns_fallback_has_expected_shape():
    """spy_annual_returns() fallback works offline and covers major market events."""
    spy = st.spy_annual_returns(cache_dir=None)
    assert len(spy) >= 10
    # 2008 should be a big loser
    assert spy.get(2008, 0.0) < -0.30
    # 2013 should be a big winner
    assert spy.get(2013, 0.0) > 0.25


def test_with_spy_benchmark_adds_columns(null_panel):
    df, truth = null_panel
    df = df.rename(columns={"ret_gross": "ret_annual", "rank": "mcap_rank"})
    annual = st.annual_portfolio_returns(df, top_n=truth["top_n"], col="ret_annual")
    spy = st.spy_annual_returns(cache_dir=None)
    enriched = st.with_spy_benchmark(annual, spy)
    assert "spy_ret" in enriched.columns
    assert "topn_vs_spy" in enriched.columns
    assert np.allclose(
        enriched["topn_vs_spy"].dropna(),
        (enriched["topn_ret"] - enriched["spy_ret"]).dropna(),
    )
