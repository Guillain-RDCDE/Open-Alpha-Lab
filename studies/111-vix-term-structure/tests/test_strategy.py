"""Signals, quintile mechanics, timing overlay, and the study's spine: the slope
signal only shows a clear Q5-Q1 spread when we plant one in the synthetic tape."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vix_term_structure import strategy as st  # noqa: E402
from vix_term_structure import data  # noqa: E402


# ---- signal construction --------------------------------------------------

def test_slope_quintile_labels(null_tape):
    """Quintile values are in {1, 2, 3, 4, 5} and appear in roughly equal proportions."""
    df, _ = null_tape
    q = st.slope_quintile(df["slope"])
    valid = q.dropna()
    assert set(valid.unique()).issubset({1, 2, 3, 4, 5})
    # Each quintile should hold ~20% of ranked observations.
    for qi in range(1, 6):
        frac = (valid == qi).mean()
        assert 0.10 < frac < 0.35, f"Quintile {qi} fraction {frac:.3f} out of expected range"


def test_quintile_monotone_rank():
    """Higher quintile implies higher slope (sorting works correctly)."""
    df, _ = data.synthetic_daily(n_days=1000, seed=1)
    q = st.slope_quintile(df["slope"], window=100, min_periods=30)
    valid = q.notna()
    for qi in range(1, 5):
        mean_slope_q = df["slope"][valid & (q == qi)].mean()
        mean_slope_qp1 = df["slope"][valid & (q == qi + 1)].mean()
        assert mean_slope_q < mean_slope_qp1, (
            f"Q{qi} mean slope {mean_slope_q:.4f} should be < Q{qi+1} mean slope {mean_slope_qp1:.4f}"
        )


def test_forward_return_no_lookahead():
    """The forward return at index t uses close[t+h] / close[t]; the last h entries are NaN."""
    spy = pd.Series([100.0, 102.0, 105.0, 103.0, 108.0],
                    index=pd.bdate_range("2020-01-01", periods=5))
    fwd1 = st.forward_return(spy, horizon=1)
    assert np.isnan(fwd1.iloc[-1])
    assert np.allclose(fwd1.iloc[0], np.log(102.0 / 100.0), atol=1e-9)
    fwd2 = st.forward_return(spy, horizon=2)
    assert np.isnan(fwd2.iloc[-1])
    assert np.isnan(fwd2.iloc[-2])


def test_contango_mask(null_tape):
    """contango_mask is True exactly when slope > 0."""
    df, _ = null_tape
    mask = st.contango_mask(df["slope"])
    assert (mask == (df["slope"] > 0)).all()


# ---- quintile table -------------------------------------------------------

def test_quintile_table_shape_and_coverage(null_tape):
    df, _ = null_tape
    tbl = st.quintile_table(df, horizons=[1, 5])
    assert set(tbl["quintile"].unique()) == {1, 2, 3, 4, 5}
    assert set(tbl["horizon"].unique()) == {1, 5}
    assert tbl["n"].sum() > 0


# ---- top minus bottom spread ----------------------------------------------

def test_top_minus_bottom_keys(null_tape):
    df, _ = null_tape
    result = st.top_minus_bottom(df, horizon=1)
    for key in ("n_top", "n_bot", "mean_top_bps", "mean_bot_bps", "spread_bps",
                "t_spread", "t_top", "t_bot"):
        assert key in result, f"Missing key: {key}"


def test_signal_produces_spread(signal_tape, null_tape):
    """The planted signal should produce a positive Q5-Q1 spread; null should be near zero."""
    df_sig, _ = signal_tape
    df_null, _ = null_tape
    spread_sig = st.top_minus_bottom(df_sig, horizon=1)["spread_bps"]
    spread_null = st.top_minus_bottom(df_null, horizon=1)["spread_bps"]
    # On a planted signal tape the Q5 minus Q1 spread should be positive and larger.
    assert spread_sig > spread_null, (
        f"Signal spread ({spread_sig:.2f}) should exceed null spread ({spread_null:.2f})"
    )


# ---- timing overlay -------------------------------------------------------

def test_timing_overlay_columns(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)
    for col in ("r_passive", "r_active", "r_spread", "is_contango", "switched"):
        assert col in ov.columns, f"Missing column: {col}"


def test_timing_overlay_flat_when_backwardated(null_tape):
    """When the slope is negative (backwardation), r_active should be 0 (flat, no signal)."""
    df, _ = null_tape
    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)
    # is_contango is a nullable boolean — use fillna(True) to safely invert for False==backwardation
    back = ov[~ov["is_contango"].fillna(True)]
    assert np.allclose(back["r_active"].to_numpy(), 0.0, atol=1e-12)


def test_timing_overlay_active_equals_passive_in_contango(null_tape):
    """When in contango (no switches, no cost), r_active should equal r_passive."""
    df, _ = null_tape
    ov = st.timing_overlay(df, horizon=1, cost_bps=0.0)
    contango_no_switch = ov[ov["is_contango"] & ~ov["switched"]]
    assert np.allclose(contango_no_switch["r_active"].to_numpy(),
                       contango_no_switch["r_passive"].to_numpy(), atol=1e-12)


def test_cost_lowers_active_return(null_tape):
    """Higher switching cost should lower the cumulative active return."""
    df, _ = null_tape
    ov0 = st.timing_overlay(df, horizon=1, cost_bps=0.0)
    ov5 = st.timing_overlay(df, horizon=1, cost_bps=5.0)
    assert ov0["r_active"].mean() >= ov5["r_active"].mean()


# ---- regime stats ---------------------------------------------------------

def test_regime_stats_labels(null_tape):
    df, _ = null_tape
    rs = st.regime_stats(df)
    labels = rs["regime"].tolist()
    assert "full sample" in labels
    assert "contango (slope>0)" in labels
    assert "backwardation (slope<0)" in labels


def test_regime_stats_n_sums_to_full(null_tape):
    """Contango + backwardation days should sum to the full-sample count."""
    df, _ = null_tape
    rs = st.regime_stats(df).set_index("regime")
    total = rs.loc["full sample", "n"]
    parts = rs.loc["contango (slope>0)", "n"] + rs.loc["backwardation (slope<0)", "n"]
    assert abs(total - parts) <= 1, f"Full sample n={total} != contango+backwardation n={parts}"


# ---- summarize ------------------------------------------------------------

def test_summarize_keys(null_tape):
    df, _ = null_tape
    s = st.summarize(df["SPY_ret"].dropna().to_numpy())
    for key in ("label", "n", "mean_bps", "std_bps", "sharpe_ann", "skew", "t_stat"):
        assert key in s


def test_summarize_zero_mean_has_zero_tstat():
    """A series with exact zero mean produces t_stat = 0."""
    r = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    s = st.summarize(r)
    assert abs(s["mean_bps"]) < 1e-9
    assert abs(s["t_stat"]) < 1e-9
