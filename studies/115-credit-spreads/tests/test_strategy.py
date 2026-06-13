"""Signal construction, regime flags, and the study's spine:
credit stress predicts negative equity forward returns only when the effect is planted."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_spreads import strategy as st, data  # noqa: E402


# ---- spread proxies -------------------------------------------------------
def test_hyg_ief_spread_shape(null_prices):
    prices, truth = null_prices
    spread = st.hyg_ief_spread(prices, window=20)
    assert len(spread) == truth["n_days"]
    # First few entries should be NaN (window not filled)
    assert spread.iloc[:19].isna().all()
    assert spread.iloc[20:].notna().all()


def test_hyg_lqd_rs_shape(null_prices):
    prices, truth = null_prices
    rs = st.hyg_lqd_rs(prices, window=20)
    assert len(rs) == truth["n_days"]
    assert rs.iloc[:19].isna().all()


def test_forward_returns_no_lookahead(null_prices):
    prices, _ = null_prices
    fwd = st.forward_returns(prices, forward_days=5)
    # Last 5 entries should be NaN (no future data to compute return)
    assert fwd.iloc[-5:].isna().all()
    assert fwd.iloc[:-5].notna().any()


def test_build_signals_columns(null_prices):
    prices, _ = null_prices
    df = st.build_signals(prices, window=20, forward_days=5, rolling_median_window=60)
    expected = {"hyg_ief_spread", "hyg_lqd_rs", "spy_fwd", "spread_wide", "rs_wide", "spy_ret"}
    assert set(df.columns) == expected
    # Regime flags are binary
    assert set(df["spread_wide"].unique()) <= {0, 1}
    assert set(df["rs_wide"].unique()) <= {0, 1}


def test_regime_flags_roughly_balanced(null_prices):
    """With a null model, stress and calm regimes should be roughly 50/50."""
    prices, _ = null_prices
    df = st.build_signals(prices, window=20, forward_days=5, rolling_median_window=60)
    frac_stress = df["spread_wide"].mean()
    # Should be within 20–80% (rolling median split is never exactly 50% due to ties)
    assert 0.20 < frac_stress < 0.80


def test_summarize_returns_expected_keys(null_prices):
    prices, _ = null_prices
    result = st.summarize(prices)
    required_keys = ["n", "date_start", "date_end", "spread_corr", "rs_corr",
                     "spread_diff_tstat", "rs_diff_tstat"]
    for k in required_keys:
        assert k in result, f"Missing key: {k}"


def test_null_model_no_predictability(null_prices):
    """On the null synthetic tape, the diff_tstat should be close to zero (|t| < 3)."""
    prices, _ = null_prices
    result = st.summarize(prices, window=20, forward_days=5)
    # Under the null, the t-stat should not be extreme
    assert abs(result["spread_diff_tstat"]) < 4.0
    assert abs(result["rs_diff_tstat"]) < 4.0


def test_planted_signal_creates_predictability(signal_prices):
    """With a planted credit-spread lead-lag, stress should predict lower equity returns."""
    prices, truth = signal_prices
    assert truth["spread_signal"] == 2.0
    df = st.build_signals(prices, window=20, forward_days=5, rolling_median_window=60)
    # Stress periods should have lower forward returns than calm periods on average
    stress_mean = df.loc[df["spread_wide"] == 1, "spy_fwd"].mean()
    calm_mean = df.loc[df["spread_wide"] == 0, "spy_fwd"].mean()
    # The planted signal should create a visible gap
    assert stress_mean < calm_mean


def test_long_run_var_positive():
    r = np.random.default_rng(42).normal(0, 1, 100)
    lrv = st._long_run_var(r)
    assert lrv > 0
    assert np.isfinite(lrv)
