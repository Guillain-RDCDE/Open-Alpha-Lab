"""Signals, quantile sort, OLS slope, shuffle null, and the study's spine:
the high-SI minus low-SI spread tracks the planted slope -- negative when
high-SI is set to bleed, positive when set to bounce, ~zero on the null."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_interest import data, strategy as st  # noqa: E402

CACHE_PATH = data.FORWARD_CACHE


# ---------------------------------------------------------------------------
# Quantile spread
# ---------------------------------------------------------------------------
def test_quantile_spread_keys(null_panel):
    df, _ = null_panel
    qs = st.quantile_spread(df, q=0.20)
    assert set(["high", "low", "spread", "tstat", "n_high", "n_low", "n"]).issubset(qs)


def test_quantile_spread_equals_high_minus_low(null_panel):
    df, _ = null_panel
    qs = st.quantile_spread(df, q=0.20)
    assert abs(qs["spread"] - (qs["high"] - qs["low"])) < 1e-12


def test_quantile_spread_deterministic(bleed_panel):
    df, _ = bleed_panel
    a = st.quantile_spread(df, q=0.20)
    b = st.quantile_spread(df, q=0.20)
    assert a == b


# ---------------------------------------------------------------------------
# OLS slope
# ---------------------------------------------------------------------------
def test_slope_keys(null_panel):
    df, _ = null_panel
    sl = st.cross_sectional_slope(df)
    assert set(["slope", "tstat", "r2", "n"]).issubset(sl)


def test_slope_sign_matches_plant_bleed(bleed_panel):
    df, _ = bleed_panel
    sl = st.cross_sectional_slope(df)
    assert sl["slope"] < 0, "Planted negative slope must yield negative OLS slope"
    assert sl["tstat"] < -2.0, f"Planted slope should be significant, got t={sl['tstat']:.2f}"


def test_slope_sign_matches_plant_bounce(bounce_panel):
    df, _ = bounce_panel
    sl = st.cross_sectional_slope(df)
    assert sl["slope"] > 0, "Planted positive slope must yield positive OLS slope"
    assert sl["tstat"] > 2.0, f"Planted slope should be significant, got t={sl['tstat']:.2f}"


# ---------------------------------------------------------------------------
# The spine: the plant knob switches the direction of the spread
# ---------------------------------------------------------------------------
def test_bleed_panel_spread_negative(bleed_panel):
    df, _ = bleed_panel
    qs = st.quantile_spread(df, q=0.20)
    assert qs["spread"] < 0.0, f"High-SI should bleed, got spread={qs['spread']:.4f}"
    assert qs["tstat"] < -2.0, f"Spread should be significant, got t={qs['tstat']:.2f}"


def test_bounce_panel_spread_positive(bounce_panel):
    df, _ = bounce_panel
    qs = st.quantile_spread(df, q=0.20)
    assert qs["spread"] > 0.0, f"High-SI should bounce, got spread={qs['spread']:.4f}"
    assert qs["tstat"] > 2.0, f"Spread should be significant, got t={qs['tstat']:.2f}"


def test_null_panel_spread_insignificant(null_panel):
    df, _ = null_panel
    qs = st.quantile_spread(df, q=0.20)
    assert np.isfinite(qs["spread"])
    assert np.isfinite(qs["tstat"])
    assert abs(qs["tstat"]) < 2.5, f"Null spread should be insignificant, got t={qs['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Shuffle null
# ---------------------------------------------------------------------------
def test_shuffle_null_keys(null_panel):
    df, _ = null_panel
    sh = st.shuffle_null(df, q=0.20, n_shuffles=300, seed=1)
    assert set(["obs", "p_value", "null_mean", "null_std"]).issubset(sh)


def test_shuffle_null_centered(null_panel):
    """Under the null the shuffle distribution is centred near zero."""
    df, _ = null_panel
    sh = st.shuffle_null(df, q=0.20, n_shuffles=500, seed=2)
    assert abs(sh["null_mean"]) < 0.01, f"Null shuffle mean too far from 0: {sh['null_mean']:.4f}"


def test_shuffle_null_rejects_plant(bleed_panel):
    """A strongly planted slope should produce a small shuffle p-value."""
    df, _ = bleed_panel
    sh = st.shuffle_null(df, q=0.20, n_shuffles=1000, seed=3)
    assert sh["p_value"] < 0.05, f"Plant should reject the null, got p={sh['p_value']:.3f}"


def test_shuffle_null_deterministic(bleed_panel):
    df, _ = bleed_panel
    a = st.shuffle_null(df, q=0.20, n_shuffles=200, seed=5)
    b = st.shuffle_null(df, q=0.20, n_shuffles=200, seed=5)
    assert a == b


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def test_bootstrap_ci_brackets_point(bleed_panel):
    df, _ = bleed_panel
    bc = st.bootstrap_spread_ci(df, q=0.20, n_boot=500, seed=7)
    assert bc["ci_low"] <= bc["point"] <= bc["ci_high"]


def test_bootstrap_ci_deterministic(null_panel):
    df, _ = null_panel
    a = st.bootstrap_spread_ci(df, q=0.20, n_boot=300, seed=9)
    b = st.bootstrap_spread_ci(df, q=0.20, n_boot=300, seed=9)
    assert a == b


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def test_cost_adjusted_spread_charges_cost(bleed_panel):
    df, _ = bleed_panel
    ca = st.cost_adjusted_spread(df, q=0.20, one_way_bps=25.0, borrow_ann_pct=8.0, horizon_months=3)
    assert ca["cost"] > 0
    # net magnitude must be smaller than gross magnitude
    assert abs(ca["net"]) <= abs(ca["gross"]) + 1e-12


def test_cost_borrow_heavier_when_high_si_is_short(bleed_panel):
    """When the spread is negative (short the crowded high-SI leg), borrow is heavier."""
    df, _ = bleed_panel
    qs = st.quantile_spread(df, q=0.20)
    assert qs["spread"] < 0
    ca = st.cost_adjusted_spread(df, q=0.20, one_way_bps=25.0, borrow_ann_pct=8.0, horizon_months=3)
    # 3m of 8%/yr borrow = 2% just for borrow, plus trade cost
    assert ca["cost"] > 0.02


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def test_summarize_returns_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(40) * 0.05)
    s = st.summarize_returns(r)
    assert set(["mean", "vol", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_returns_short_series():
    s = st.summarize_returns(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="si_forward.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_spread_is_finite():
    panel = data.real_panel(horizon_col="fwd_3m", cache_path=CACHE_PATH)
    qs = st.quantile_spread(panel, q=0.20)
    assert np.isfinite(qs["spread"])
    assert isinstance(qs["tstat"], float)


@requires_cache
def test_real_slope_is_finite():
    panel = data.real_panel(horizon_col="fwd_3m", cache_path=CACHE_PATH)
    sl = st.cross_sectional_slope(panel)
    assert np.isfinite(sl["slope"])
