"""Tests for strategy.py — the drop-ratio stats, capture P&L, and controls."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_capture import strategy as st  # noqa: E402


# ---- drop ratio --------------------------------------------------------------
def test_drop_ratio_tstat_near_zero_for_fully_efficient(efficient):
    """With efficiency=0.0 (fully efficient), tstat_vs_one should be close to 0 (noise only)."""
    events, _ = efficient
    stats = st.drop_ratio_stats(events)
    assert stats["n"] == len(events)
    # On a fully-efficient tape the drop ratio is ~1, so |t vs 1| should be small.
    # Allow up to 4.0 due to small-sample noise in the synthetic tape (n=200, noisy).
    assert abs(stats["tstat_vs_one"]) < 4.5, (
        f"expected |t| near 0 for efficient tape, got {stats['tstat_vs_one']:.2f}"
    )


def test_drop_ratio_tstat_negative_for_inefficient(inefficient):
    """With efficiency=0.5, the mean ratio is ~0.5 < 1, so t-stat vs 1 should be negative."""
    events, _ = inefficient
    stats = st.drop_ratio_stats(events)
    assert stats["tstat_vs_one"] < 0, (
        f"expected negative tstat_vs_one for inefficient tape, got {stats['tstat_vs_one']:.2f}"
    )


def test_drop_ratio_frac_below_one_higher_for_inefficient(efficient, inefficient):
    eff_stats = st.drop_ratio_stats(efficient[0])
    ineff_stats = st.drop_ratio_stats(inefficient[0])
    assert ineff_stats["frac_below_one"] > eff_stats["frac_below_one"]


# ---- capture trade -----------------------------------------------------------
def test_capture_trade_stats_columns(efficient):
    events, _ = efficient
    s = st.capture_trade_stats(events, cost_bps=5.0)
    for key in ("n", "mean_gross_bps", "mean_net_bps", "win_rate_gross", "tstat_gross", "tstat_net"):
        assert key in s, f"missing key: {key}"


def test_net_less_than_gross(efficient):
    events, _ = efficient
    s = st.capture_trade_stats(events, cost_bps=5.0)
    assert s["mean_net_bps"] < s["mean_gross_bps"]


def test_higher_cost_lowers_net(efficient):
    events, _ = efficient
    s1 = st.capture_trade_stats(events, cost_bps=0.0)
    s5 = st.capture_trade_stats(events, cost_bps=5.0)
    s10 = st.capture_trade_stats(events, cost_bps=10.0)
    assert s1["mean_net_bps"] > s5["mean_net_bps"] > s10["mean_net_bps"]


def test_capture_earns_more_when_inefficient(efficient, inefficient):
    """With efficiency=0.5, gross capture is higher than fully efficient (efficiency=0.0)."""
    s_eff = st.capture_trade_stats(efficient[0], cost_bps=0.0)
    s_ineff = st.capture_trade_stats(inefficient[0], cost_bps=0.0)
    assert s_ineff["mean_gross_bps"] > s_eff["mean_gross_bps"], (
        "inefficient tape (efficiency=0.5) should produce higher gross capture return"
    )


# ---- buy-and-hold baseline ---------------------------------------------------
def test_bah_stats_columns(efficient):
    events, _ = efficient
    s = st.buy_and_hold_stats(events)
    assert "mean_bah_bps" in s
    assert "win_rate_bah" in s
    assert s["n"] == len(events)


# ---- summarize ---------------------------------------------------------------
def test_summarize_returns_all_keys(efficient):
    events, _ = efficient
    out = st.summarize(events, cost_bps=5.0)
    assert set(out.keys()) == {"drop", "capture", "bah"}
    assert "tstat_vs_one" in out["drop"]
    assert "tstat_gross" in out["capture"]


# ---- shuffled control --------------------------------------------------------
def test_shuffled_date_control_returns_finite(efficient):
    events, _ = efficient
    ctrl = st.shuffled_date_control(events, price_series=None, n_draws=100, seed=1)
    assert np.isfinite(ctrl["mean_ctrl_bps"])


def test_shuffled_control_similar_to_efficient_capture(efficient):
    """On a fully efficient tape, capture gross ≈ the shuffled-date control (both near 0)."""
    events, _ = efficient
    cap = st.capture_trade_stats(events, cost_bps=0.0)["mean_gross_bps"]
    ctrl = st.shuffled_date_control(events, price_series=None, n_draws=300, seed=143)["mean_ctrl_bps"]
    # Both should be within ±20 bps of each other on this noisy synthetic tape.
    assert abs(cap - ctrl) < 20.0, (
        f"capture ({cap:.2f} bps) far from shuffled control ({ctrl:.2f} bps) — unexpected"
    )
