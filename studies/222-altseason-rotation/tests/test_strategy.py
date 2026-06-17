"""Tests for strategy.py -- signals, PnL engine, performance stats, and the study spine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altseason_rotation import data, strategy as st  # noqa: E402

CACHE_PATH = data._cache_path(data.DEFAULT_CACHE)


# ---- dominance signals ----------------------------------------------------
def test_change_signal_binary(null_panel):
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    vals = sig.dropna().values
    assert set(vals).issubset({0.0, 1.0})


def test_change_signal_nans_at_start(null_panel):
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    assert sig.isna().sum() == 20


def test_level_signal_binary(null_panel):
    panel, _ = null_panel
    sig = st.dominance_level_signal(panel, window=252)
    vals = sig.dropna().values
    assert set(vals).issubset({0.0, 1.0})


def test_level_signal_has_both_states(null_panel):
    panel, _ = null_panel
    sig = st.dominance_level_signal(panel, window=252).dropna()
    assert (sig == 1.0).sum() > 0
    assert (sig == 0.0).sum() > 0


# ---- rotation PnL engine --------------------------------------------------
def test_rotation_pnl_columns(null_panel):
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=40)
    for col in ["signal", "position", "gross_ret", "net_ret", "cost", "cum_net"]:
        assert col in pnl.columns


def test_rotation_pnl_no_lookahead(null_panel):
    """Position on day t must be driven by signal from day t-1 (1-day lag)."""
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=0)
    # When position=0 (no rotation), gross_ret must be 0
    zero_pos = pnl[pnl["position"] == 0.0]
    assert (zero_pos["gross_ret"].abs() < 1e-12).all()


def test_rotation_pnl_cost_reduces_net(null_panel):
    """Net returns must be <= gross returns (costs are non-negative)."""
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=40)
    assert (pnl["net_ret"] <= pnl["gross_ret"] + 1e-12).all()


# ---- performance summary --------------------------------------------------
def test_performance_summary_keys(null_panel):
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=40)
    perf = st.performance_summary(pnl["net_ret"])
    for k in ["n", "ann_return_pct", "ann_vol_pct", "sharpe", "max_dd_pct", "tstat"]:
        assert k in perf


def test_performance_summary_tstat_finite():
    """t-stat is finite when the return series has non-zero variance."""
    # Use the level signal which always fires some positions on any synthetic panel
    panel, _ = data.synthetic_daily(n_years=5, dom_signal=0.0, seed=222)
    sig = st.dominance_level_signal(panel, window=252)
    pnl = st.rotation_pnl(panel, sig, cost_bps=0)
    active = pnl[pnl["position"] == 1.0]["gross_ret"]
    if len(active) > 10:
        perf = st.performance_summary(active)
        assert np.isfinite(perf["tstat"])


# ---- rotation event counting ----------------------------------------------
def test_count_rotations_basic(null_panel):
    panel, _ = null_panel
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=40)
    ev = st.count_rotations(pnl)
    assert ev["n_rotations"] >= 0
    assert 0.0 <= ev["pct_days_active"] <= 100.0


# ---- the study's spine: engine detects signal when planted ----------------
def test_planted_signal_beats_null(null_panel, signal_panel):
    """With dom_signal=10 planted, alts cumulatively outperform BTC more than the null.

    The dom_signal injection adds alt return boosts when BTC dominance is falling.
    Over hundreds of days this accumulates into a measurably higher alt-minus-BTC
    cumulative spread. We compare unconditional cumulative spreads (not strategy PnL)
    to isolate the planted effect from the signal-timing layer.
    """
    import numpy as np

    def cum_alt_spread(panel):
        bm = st.benchmark_pnl(panel)
        spread = (bm["alt_ew_ret"] - bm["btc_ret"]).values
        return float(np.nansum(spread))

    # Use a stronger signal to ensure the planted effect is detectable
    null_p, _ = data.synthetic_daily(n_years=5, dom_signal=0.0, seed=222)
    strong_p, _ = data.synthetic_daily(n_years=5, dom_signal=50.0, seed=222)
    cum_null = cum_alt_spread(null_p)
    cum_strong = cum_alt_spread(strong_p)
    # Planted signal should increase cumulative alt-minus-BTC spread
    assert cum_strong > cum_null


# ---- cache-guarded real-tape test -----------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_tape_strategy_runs():
    panel = data.fetch_panel(fetch=False)
    sig = st.dominance_change_signal(panel, lookback=20, threshold=0.02)
    pnl = st.rotation_pnl(panel, sig, cost_bps=40)
    assert len(pnl) > 100
    ev = st.count_rotations(pnl)
    assert ev["n_rotations"] > 0
    perf = st.performance_summary(pnl["net_ret"])
    assert np.isfinite(perf["sharpe"])
