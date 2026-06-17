"""Tests for Study 241 (Buy-the-Dip).

Five offline tests using synthetic tapes only. No real-tape or cache dependency.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buy_the_dip import data, strategy as st  # noqa: E402

CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "bars_SPY_daily.parquet")
)


# ---------------------------------------------------------------------------
# Test 1: Drawdown computation is non-positive and correct at ATH
# ---------------------------------------------------------------------------
def test_running_drawdown_nonpositive(random_walk):
    """Running drawdown must be <= 0 at all times (definition)."""
    bars, _ = random_walk
    dd = st.running_drawdown(bars["close"])
    assert (dd <= 1e-9).all(), "Drawdown must be non-positive"
    # At the very first bar, we're at the first day's price (not necessarily ATH
    # if we define ATH as expanding max of close). First bar is always at ATH.
    assert abs(float(dd.iloc[0])) < 1e-9, "First bar should be at 0 drawdown"


# ---------------------------------------------------------------------------
# Test 2: Equity curves start at 1.0 and are monotone in cash-earning variant
# ---------------------------------------------------------------------------
def test_equity_curves_start_at_one(random_walk):
    """Both buy-and-hold and dip-buyer equity curves must start at 1.0."""
    bars, _ = random_walk
    bh = st.buy_and_hold_equity(bars)
    eq, _ = st.dip_buyer_equity(bars, threshold=0.05, cost_bps=0.0, cash_rate_annual=0.0)
    assert abs(float(bh.iloc[0]) - 1.0) < 1e-6, "BH equity must start at 1.0"
    assert abs(float(eq.iloc[0]) - 1.0) < 1e-6, "Dip-buyer equity must start at 1.0"


# ---------------------------------------------------------------------------
# Test 3: Cash drag — dip-buyer underperforms BH in a secular bull market
# ---------------------------------------------------------------------------
def test_cash_drag_bull_market(random_walk):
    """In a positive-drift GBM, dip-buying at 10% threshold underperforms BH.

    This is the spine of the study: the folk rule fails in bull markets because
    cash earns zero while waiting. We test with a long 10-year synthetic tape
    with strong positive drift to make this reliable.
    """
    # Use a tape with strong positive drift so the result is clear
    bars, _ = data.synthetic_daily(
        n_days=2520, drift=0.0008, daily_vol=0.010, reversion=0.0, seed=999
    )
    bh = st.buy_and_hold_equity(bars)
    eq, meta = st.dip_buyer_equity(bars, threshold=0.10, cost_bps=0.0, cash_rate_annual=0.0)
    bh_final = float(bh.iloc[-1])
    eq_final = float(eq.iloc[-1])
    # The dip-buyer must spend some time in cash (fraction_invested < 1)
    assert meta["fraction_invested"] < 1.0, "Dip-buyer should not always be invested"
    # In a bull market with zero cash rate, BH should outperform
    # (this is the whole point of the study)
    assert bh_final > eq_final, "Buy-and-hold should beat dip-buying in a bull market (cash drag)"


# ---------------------------------------------------------------------------
# Test 4: Cash drag dominates — higher threshold means lower fraction invested
# ---------------------------------------------------------------------------
def test_higher_threshold_lower_fraction_invested(random_walk):
    """A higher dip threshold means more time in cash and fewer trades.

    This confirms the cash-drag mechanism: stricter thresholds reduce time
    in market, which is the primary driver of underperformance in bull markets.
    """
    bars, _ = random_walk
    thresholds = [0.02, 0.05, 0.10, 0.20]
    fracs = []
    n_buys_list = []
    for thr in thresholds:
        _, meta = st.dip_buyer_equity(bars, threshold=thr, cost_bps=0.0, cash_rate_annual=0.0)
        fracs.append(meta["fraction_invested"])
        n_buys_list.append(meta["n_buys"])

    # Higher threshold => lower (or equal) fraction invested
    for i in range(len(fracs) - 1):
        assert fracs[i] >= fracs[i + 1], (
            f"Threshold {thresholds[i]} should have >= frac than {thresholds[i+1]}: "
            f"{fracs[i]:.3f} vs {fracs[i+1]:.3f}"
        )
    # Higher threshold => fewer (or equal) buys
    for i in range(len(n_buys_list) - 1):
        assert n_buys_list[i] >= n_buys_list[i + 1], (
            f"Threshold {thresholds[i]} should have >= buys than {thresholds[i+1]}: "
            f"{n_buys_list[i]} vs {n_buys_list[i+1]}"
        )


# ---------------------------------------------------------------------------
# Test 5: Threshold sweep — summarize_equity produces consistent output
# ---------------------------------------------------------------------------
def test_threshold_sweep_shape(random_walk):
    """threshold_sweep returns a DataFrame with expected columns and one row per threshold."""
    bars, _ = random_walk
    thresholds = [0.05, 0.10, 0.20]
    df = st.threshold_sweep(bars, thresholds=thresholds, cost_bps=0.0, cash_rate_annual=0.0)
    expected_cols = {
        "threshold_pct", "cagr", "bh_cagr", "cagr_vs_bh",
        "sharpe", "bh_sharpe", "max_dd", "n_buys", "frac_invested",
    }
    assert len(df) == len(thresholds), "One row per threshold"
    assert expected_cols.issubset(set(df.columns)), "Missing expected columns"
    # All CAGR values should be finite
    assert df["cagr"].notna().all(), "CAGR values must be finite"
    assert df["bh_cagr"].notna().all(), "BH CAGR must be finite"
    # BH CAGR is the same for all thresholds (same tape)
    assert df["bh_cagr"].nunique() == 1, "BH CAGR should be identical for all thresholds"


# ---------------------------------------------------------------------------
# Test 6 (cache-gated): real SPY tape — every threshold underperforms BH
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="real-tape cache absent offline/CI",
)
def test_real_tape_all_thresholds_underperform():
    """On real SPY data, every dip threshold underperforms buy-and-hold (cash drag)."""
    bars = data.fetch_daily("SPY", fetch=False)
    df = st.threshold_sweep(bars, cost_bps=1.0, cash_rate_annual=0.0)
    # Every threshold should show negative CAGR vs BH
    for _, row in df.iterrows():
        assert row["cagr_vs_bh"] < 0, (
            f"Threshold {row['threshold_pct']}%: expected underperformance "
            f"(cagr_vs_bh={row['cagr_vs_bh']:.4f})"
        )
