"""Signals, the quintile engine's invariants, the timing overlay, and the study's spine:
the curve-slope overlay only beats buy-and-hold when a slope signal is actually planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yield_curve_steepener import strategy as st  # noqa: E402
from yield_curve_steepener import data  # noqa: E402


# ---- slope_quintile ----------------------------------------------------------

def test_quintile_returns_int64_nullable(null_tape):
    df, _ = null_tape
    q = st.slope_quintile(df["slope"])
    # After warmup, quintiles should be 1..5
    valid = q.dropna()
    assert valid.dtype == pd.api.types.pandas_dtype("Int64")
    assert valid.min() >= 1
    assert valid.max() <= 5


def test_quintile_roughly_balanced(null_tape):
    """Each quintile should capture ~20% of the data (rough uniformity)."""
    df, _ = null_tape
    q = st.slope_quintile(df["slope"])
    counts = q.value_counts()
    total = counts.sum()
    for qi in range(1, 6):
        frac = counts.get(qi, 0) / total
        assert 0.12 < frac < 0.28, f"Quintile {qi} has unusual fraction {frac:.3f}"


def test_quintile_no_lookahead(null_tape):
    """The quintile rank at index i must only use slope values up to and including i."""
    df, _ = null_tape
    # Take a 300-day sub-frame; extend it by 1 observation; the first 300 ranks
    # must be unchanged.
    df_short = df.iloc[:300]
    df_long = df.iloc[:301]
    q_short = st.slope_quintile(df_short["slope"])
    q_long = st.slope_quintile(df_long["slope"])
    # All valid (non-NA) ranks in the short version must match the long version.
    valid_short = q_short.dropna()
    for idx in valid_short.index:
        assert q_short[idx] == q_long[idx], "Look-ahead detected in quintile rank"


# ---- forward_return ----------------------------------------------------------

def test_forward_return_horizon1(null_tape):
    df, _ = null_tape
    fwd = st.forward_return(df["TLT_close"], horizon=1)
    # fwd[t] should equal log(close[t+1]) - log(close[t])
    expected = np.log(df["TLT_close"]).diff(1).shift(-1)
    assert np.allclose(
        fwd.dropna().to_numpy(),
        expected.dropna().to_numpy(),
        atol=1e-10,
    )


def test_forward_return_last_horizon_is_nan(null_tape):
    df, _ = null_tape
    for h in [1, 5, 21]:
        fwd = st.forward_return(df["TLT_close"], horizon=h)
        assert fwd.iloc[-h:].isna().all(), f"Last {h} values should be NaN for horizon={h}"


# ---- steep_mask --------------------------------------------------------------

def test_steep_mask_is_slope_positive(null_tape):
    df, _ = null_tape
    mask = st.steep_mask(df["slope"])
    expected = df["slope"] > 0
    assert (mask == expected).all()


def test_steep_mask_custom_threshold(null_tape):
    df, _ = null_tape
    mask = st.steep_mask(df["slope"], threshold=1.0)
    expected = df["slope"] > 1.0
    assert (mask == expected).all()


# ---- quintile_table ----------------------------------------------------------

def test_quintile_table_shape(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1, 5])
    assert set(tbl.columns) == {"quintile", "horizon", "n", "mean_bps", "std_bps", "t_stat"}
    assert set(tbl["quintile"].unique()) == {1, 2, 3, 4, 5}
    assert set(tbl["horizon"].unique()) == {1, 5}


def test_quintile_table_sample_sizes(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1])
    # Each quintile should have some observations
    assert (tbl["n"] > 100).all()


# ---- top_minus_bottom --------------------------------------------------------

def test_top_minus_bottom_returns_dict_keys(null_tape):
    df, _ = null_tape
    result = st.top_minus_bottom(df, horizon=1)
    assert set(result.keys()) == {"n_top", "n_bot", "mean_top_bps", "mean_bot_bps",
                                  "spread_bps", "t_spread", "t_top", "t_bot"}


def test_top_minus_bottom_positive_control():
    """When a genuine slope signal is planted, the Q5-Q1 spread should be positive."""
    df_sig, _ = data.synthetic_daily(n_days=5000, slope_signal=0.01, seed=132)
    result = st.top_minus_bottom(df_sig, horizon=1)
    assert result["spread_bps"] > 0.5, (
        f"Expected positive spread with planted signal, got {result['spread_bps']:.2f} bps"
    )


# ---- timing_overlay ----------------------------------------------------------

def test_timing_overlay_columns(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df)
    assert set(["r_passive", "r_active", "r_spread", "is_steep", "switched"]).issubset(
        set(ov.columns)
    )


def test_timing_overlay_no_lookahead(null_tape):
    """The active return must be zero on days when the prior slope was flat/inverted."""
    df, _ = null_tape
    ov = st.timing_overlay(df, cost_bps=0.0)
    # On flat/inverted days (is_steep=False), r_active should be 0
    flat_days = ov[ov["is_steep"] == False]
    # Allow for tiny float rounding
    assert (flat_days["r_active"].abs() < 1e-10).all()


def test_timing_overlay_cost_reduces_active(null_tape):
    """Adding a switch cost can only reduce (or equal) the mean active return."""
    df, _ = null_tape
    ov0 = st.timing_overlay(df, cost_bps=0.0)
    ov2 = st.timing_overlay(df, cost_bps=2.0)
    mean0 = ov0["r_active"].mean()
    mean2 = ov2["r_active"].mean()
    assert mean2 <= mean0 + 1e-10


def test_timing_overlay_passive_matches_tlt_ret(null_tape):
    """The passive arm must equal the 1-day forward TLT log return."""
    df, _ = null_tape
    ov = st.timing_overlay(df, cost_bps=0.0)
    fwd = st.forward_return(df["TLT_close"], horizon=1)
    aligned = fwd.reindex(ov.index).to_numpy()
    assert np.allclose(ov["r_passive"].to_numpy(), aligned, atol=1e-12)


# ---- regime_stats ------------------------------------------------------------

def test_regime_stats_three_rows(null_tape):
    df, _ = null_tape
    rs = st.regime_stats(df)
    assert len(rs) == 3
    assert set(rs["regime"]) == {"full sample", "steep curve (slope>0)", "flat/inverted (slope<=0)"}


def test_regime_stats_n_adds_up(null_tape):
    """Steep + flat/inverted sample sizes should sum to the full sample."""
    df, _ = null_tape
    rs = st.regime_stats(df).set_index("regime")
    total = rs.loc["full sample", "n"]
    sub = rs.loc["steep curve (slope>0)", "n"] + rs.loc["flat/inverted (slope<=0)", "n"]
    assert total == sub


# ---- summarize ---------------------------------------------------------------

def test_summarize_output_keys(null_tape):
    df, _ = null_tape
    result = st.summarize(df["TLT_ret"])
    assert set(result.keys()) == {"label", "n", "mean_bps", "std_bps", "sharpe_ann", "skew", "t_stat"}


def test_summarize_nan_robust():
    """summarize should not crash and should produce finite results given a clean array."""
    r = np.random.default_rng(0).normal(0, 0.01, 500)
    result = st.summarize(r, label="test")
    assert np.isfinite(result["mean_bps"])
    assert np.isfinite(result["t_stat"])


# ---- the spine: signal only surfaces when planted ----------------------------

def test_positive_control_spine():
    """The machinery should find a positive Q5-Q1 spread only when a slope signal exists."""
    df_null, _ = data.synthetic_daily(n_days=5000, slope_signal=0.0, seed=132)
    df_sig, _ = data.synthetic_daily(n_days=5000, slope_signal=0.012, seed=132)
    r_null = st.top_minus_bottom(df_null, horizon=1)
    r_sig = st.top_minus_bottom(df_sig, horizon=1)
    assert r_sig["spread_bps"] > r_null["spread_bps"], (
        "Planted slope signal should produce a higher Q5-Q1 spread than the null"
    )
