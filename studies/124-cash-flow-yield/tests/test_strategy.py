"""The quintile sort, IC comparison, and the study spine: OCF yield sorts find the edge
only when it is planted — and on a null panel the spread is noise."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_flow_yield import strategy as st  # noqa: E402


# ---- quintile_sort -----------------------------------------------------------

def test_quintile_sort_shape(signal_panel):
    ocfy, _, fwd, _ = signal_panel
    result = st.quintile_sort(ocfy, fwd)
    # Columns: Q1 ... Q5, hedge, market, n_firms
    expected_cols = {"Q1", "Q2", "Q3", "Q4", "Q5", "hedge", "market", "n_firms"}
    assert expected_cols.issubset(set(result.columns))
    assert len(result) > 0


def test_quintile_sort_hedge_direction_with_premium(signal_panel):
    """When a premium is planted, the Q5-Q1 hedge should be mostly positive."""
    ocfy, _, fwd, truth = signal_panel
    result = st.quintile_sort(ocfy, fwd)
    assert truth.has_ocfy_edge
    # The mean hedge return over planted years should be consistently positive.
    mean_hedge = result["hedge"].mean()
    assert mean_hedge > 0.0, (
        f"Expected positive hedge with planted premium {truth.ocfy_premium}, got {mean_hedge:.4f}"
    )


def test_quintile_sort_hedge_near_zero_on_null(null_panel):
    """On a null panel (no premium) the hedge should be centred near zero."""
    ocfy, _, fwd, truth = null_panel
    assert not truth.has_ocfy_edge
    result = st.quintile_sort(ocfy, fwd)
    mean_hedge = result["hedge"].mean()
    # With 100 firms and 15 years of noise, the hedge should be small but not exactly 0.
    assert abs(mean_hedge) < 0.04, (
        f"Hedge on null panel should be near zero; got {mean_hedge:.4f}"
    )


def test_quintile_sort_q5_gt_q1_on_signal(signal_panel):
    """On the signal panel, Q5 should on average exceed Q1."""
    ocfy, _, fwd, _ = signal_panel
    result = st.quintile_sort(ocfy, fwd)
    assert result["Q5"].mean() > result["Q1"].mean()


def test_quintile_sort_empty_on_too_few_firms():
    """If the panel has fewer than min_firms, the output should be empty."""
    import pandas as pd
    from cash_flow_yield import data
    ocfy, _, fwd, _ = data.synthetic_panel(n_firms=5, n_years=5)
    result = st.quintile_sort(ocfy, fwd, min_firms=20)
    assert len(result) == 0


# ---- hedge_vs_market --------------------------------------------------------

def test_hedge_vs_market_shape(signal_panel):
    ocfy, _, fwd, _ = signal_panel
    result = st.hedge_vs_market(ocfy, fwd)
    assert set(result.columns) >= {"long", "market", "spread"}
    assert len(result) > 0


def test_hedge_vs_market_spread_positive_with_premium(signal_panel):
    """Top-quintile long vs market should generate a positive spread under the planted premium."""
    ocfy, _, fwd, _ = signal_panel
    result = st.hedge_vs_market(ocfy, fwd)
    assert result["spread"].mean() > 0.0


# ---- pairwise_ic ------------------------------------------------------------

def test_pairwise_ic_shape(signal_panel):
    ocfy, ey, fwd, _ = signal_panel
    result = st.pairwise_ic(ocfy, ey, fwd)
    assert set(result.columns) >= {"ic_a", "ic_b", "ic_diff"}
    assert len(result) > 0


def test_pairwise_ic_values_in_range(signal_panel):
    """Spearman ICs are bounded in [-1, 1]."""
    ocfy, ey, fwd, _ = signal_panel
    result = st.pairwise_ic(ocfy, ey, fwd)
    assert (result["ic_a"].abs() <= 1.0).all()
    assert (result["ic_b"].abs() <= 1.0).all()


def test_pairwise_ic_ocfy_higher_than_ey_when_only_ocfy_planted():
    """When only OCF-yield has a premium (EY premium = 0), OCF IC should exceed EY IC on average."""
    from cash_flow_yield import data
    ocfy, ey, fwd, _ = data.synthetic_panel(
        n_firms=200, n_years=18, ocfy_premium=0.06, ey_premium=0.0, seed=77
    )
    result = st.pairwise_ic(ocfy, ey, fwd)
    assert result["ic_a"].mean() > result["ic_b"].mean()


# ---- summary ----------------------------------------------------------------

def test_summary_keys(signal_panel):
    ocfy, _, fwd, _ = signal_panel
    result = st.quintile_sort(ocfy, fwd)
    s = st.summary(result["hedge"])
    assert set(s.keys()) >= {"mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n"}


def test_summary_nan_on_too_short():
    """Summary returns NaN fields on a trivially short series (n < 2)."""
    import pandas as pd
    s = st.summary(pd.Series([0.10]))
    assert np.isnan(s["tstat"])


def test_summary_tstat_significant_on_strong_signal():
    """On a long planted-premium series the t-stat should be clearly positive."""
    from cash_flow_yield import data
    ocfy, _, fwd, _ = data.synthetic_panel(
        n_firms=300, n_years=40, ocfy_premium=0.08, ey_premium=0.0, seed=1
    )
    result = st.quintile_sort(ocfy, fwd)
    s = st.summary(result["hedge"])
    assert s["tstat"] > 1.5, (
        f"Expected strong t-stat with planted premium, got {s['tstat']:.2f}"
    )
