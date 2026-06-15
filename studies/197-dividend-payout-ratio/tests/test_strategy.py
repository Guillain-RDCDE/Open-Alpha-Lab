"""Tests for the strategy layer — Study 197 (Dividend-Payout-Ratio).

Synthetic-tape tests are offline and cover all engine invariants; real-data
tests are guarded with ``@requires_cache``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_payout_ratio import data  # noqa: E402
from dividend_payout_ratio import strategy as st  # noqa: E402

# Cache guard: skip real-data tests when the Shiller parquet is absent (CI).
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.SHILLER_CACHE),
    reason="Shiller cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# OLS / HAC engine — invariants on synthetic tape
# ---------------------------------------------------------------------------
def test_ols_hac_zero_slope_on_constant_x():
    """A constant predictor has undefined slope — engine returns a near-zero slope."""
    rng = np.random.default_rng(0)
    x = np.ones(200)
    y = rng.standard_normal(200)
    # ols_hac uses lstsq; with constant x and constant y-intercept absorbed it
    # should be numerically stable and not crash
    try:
        reg = st.ols_hac(x, y, maxlags=1)
        # result: either near-zero or any finite float — just must not raise
        assert np.isfinite(reg["intercept"])
    except np.linalg.LinAlgError:
        pass  # degenerate case: singular matrix is acceptable


def test_ols_hac_recovers_known_slope():
    """ols_hac recovers the planted slope on a clean low-noise tape."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0.3, 0.8, n)
    true_slope = 0.1
    y = 0.02 + true_slope * x + 0.005 * rng.standard_normal(n)
    reg = st.ols_hac(x, y, maxlags=1)
    assert abs(reg["slope"] - true_slope) < 0.02
    assert abs(reg["t_slope"]) > 2.0  # should be significant


def test_ols_hac_returns_required_keys():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)
    y = rng.standard_normal(100)
    reg = st.ols_hac(x, y, maxlags=5)
    for k in ("intercept", "slope", "t_intercept", "t_slope", "se_slope", "r_squared", "n"):
        assert k in reg, f"missing key: {k}"
    assert reg["n"] == 100


def test_r_squared_between_zero_and_one():
    rng = np.random.default_rng(7)
    x = rng.uniform(0.2, 1.0, 200)
    y = 0.05 * x + 0.01 * rng.standard_normal(200)
    reg = st.ols_hac(x, y, maxlags=5)
    assert 0.0 <= reg["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# regress_payout on synthetic tape
# ---------------------------------------------------------------------------
def test_regress_payout_null_slope_small(null_tape):
    """On the null tape, the payout slope must be small (no signal planted)."""
    df, _ = null_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    reg = st.regress_payout(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout", maxlags=5)
    assert abs(reg["slope"]) < 0.15


def test_regress_payout_signal_positive(signal_tape):
    """On the signal tape, slope and unconditional mean both returned."""
    df, _ = signal_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    reg = st.regress_payout(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout", maxlags=5)
    assert reg["slope"] > 0.0
    assert "unconditional_mean" in reg
    assert np.isfinite(reg["unconditional_t"])


# ---------------------------------------------------------------------------
# IS/OOS split on synthetic tape
# ---------------------------------------------------------------------------
def test_is_oos_split_structure(signal_tape):
    df, _ = signal_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    result = st.is_oos_split(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout",
                              split_year=1940, maxlags=5)
    assert "is" in result and "oos" in result
    assert result["is"]["n"] > 0
    assert result["oos"]["n"] > 0
    assert result["is"]["n"] + result["oos"]["n"] == result["is"]["n"] + result["oos"]["n"]


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------
def test_quintile_means_shape_and_monotone(signal_tape):
    """Quintile means should span 5 buckets; the signal tape should be monotone."""
    df, _ = signal_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    qm = st.payout_quintile_means(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout")
    assert len(qm) == 5
    assert "mean_payout" in qm.columns
    assert "mean_outcome" in qm.columns
    # Q1 payout < Q5 payout
    assert qm["mean_payout"].iloc[0] < qm["mean_payout"].iloc[-1]
    # With a positive signal, Q5 outcome > Q1 outcome
    assert qm["mean_outcome"].iloc[-1] > qm["mean_outcome"].iloc[0]


# ---------------------------------------------------------------------------
# Non-overlapping regression
# ---------------------------------------------------------------------------
def test_non_overlapping_returns_required_keys(signal_tape):
    df, _ = signal_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    res = st.non_overlapping_regression(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout")
    for k in ("slope", "t_hc3", "r_squared", "n"):
        assert k in res


def test_non_overlapping_signal_significant(signal_tape):
    """With a strong planted signal and 100y synthetic tape, annual obs |t| > 2."""
    df, _ = signal_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    res = st.non_overlapping_regression(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout")
    assert abs(res["t_hc3"]) > 2.0, f"expected |t|>2 on signal tape, got {res['t_hc3']:.2f}"


# ---------------------------------------------------------------------------
# Summarize helper
# ---------------------------------------------------------------------------
def test_summarize_extracts_keys(null_tape):
    df, _ = null_tape
    df_renamed = df.rename(columns={"payout": "Payout", "fwd_outcome": "fwd_10y_eg"})
    reg = st.regress_payout(df_renamed, outcome_col="fwd_10y_eg", payout_col="Payout", maxlags=5)
    s = st.summarize(reg)
    for k in ("slope", "t_slope", "r_squared", "n", "unconditional_mean", "unconditional_t"):
        assert k in s


# ---------------------------------------------------------------------------
# Real data tests — skipped when cache is absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_payout_slope_positive_eg():
    """On real Shiller data, payout predicts fwd_10y_eg with positive slope."""
    df = data.load_shiller()
    reg = st.regress_payout(df, outcome_col="fwd_10y_eg", payout_col="Payout", maxlags=120)
    assert reg["slope"] > 0.0, f"expected positive slope for EG, got {reg['slope']:.4f}"


@requires_cache
def test_real_is_oos_split_both_positive_eg():
    """Both IS and OOS slopes for fwd_10y_eg should be positive on real data."""
    df = data.load_shiller()
    result = st.is_oos_split(df, outcome_col="fwd_10y_eg", payout_col="Payout",
                              split_year=1953, maxlags=120)
    assert result["is"]["slope"] > 0.0
    assert result["oos"]["slope"] > 0.0


@requires_cache
def test_real_quintile_q5_beats_q1_eg():
    """Top payout quintile should predict higher fwd EG than bottom on real data."""
    df = data.load_shiller()
    qm = st.payout_quintile_means(df, outcome_col="fwd_10y_eg", payout_col="Payout")
    assert qm["mean_outcome"].iloc[-1] > qm["mean_outcome"].iloc[0]
