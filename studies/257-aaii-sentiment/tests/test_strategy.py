"""Signals, regime sort, regression, overlay, summary stats, and the study's spine:
the contrarian regression recovers a NEGATIVE slope only when the edge is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aaii_sentiment import data, strategy as st  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.04)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert s["n"] == 2
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    s = st.summarize(pd.Series(rng.normal(0.05, 0.03, 100)))
    assert s["tstat"] > 0


def test_annualise_monthly():
    out = st.annualise_monthly(st.summarize(pd.Series([0.01] * 60)), periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Regime sort
# ---------------------------------------------------------------------------
def test_regime_returns_labels(edge_panel):
    panel, _ = edge_panel
    reg = st.regime_returns(panel)
    assert set(reg["regime"].unique()).issubset({"low", "mid", "high"})
    assert {"spread", "ret", "regime"}.issubset(reg.columns)


def test_regime_summary_shape(null_panel):
    panel, _ = null_panel
    rs = st.regime_summary(panel)
    assert list(rs.index) == ["low", "mid", "high"]
    assert {"mean_mo", "mean_ann", "tstat", "n"}.issubset(rs.columns)


def test_regime_spread_signs(edge_panel):
    """Low regime contributes +ret, high regime contributes -ret, mid contributes 0."""
    panel, _ = edge_panel
    reg = st.regime_returns(panel)
    cs = st.regime_spread(panel)
    low_idx = reg.index[reg["regime"] == "low"]
    high_idx = reg.index[reg["regime"] == "high"]
    mid_idx = reg.index[reg["regime"] == "mid"]
    np.testing.assert_allclose(cs.loc[low_idx].values, reg.loc[low_idx, "ret"].values)
    np.testing.assert_allclose(cs.loc[high_idx].values, -reg.loc[high_idx, "ret"].values)
    np.testing.assert_allclose(cs.loc[mid_idx].values, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Predictive regression -- the spine: edge knob flips the slope sign/significance
# ---------------------------------------------------------------------------
def test_regression_recovers_negative_slope(edge_panel):
    """With a planted contrarian edge, the slope is negative and significant."""
    panel, _ = edge_panel
    reg = st.predictive_regression(panel)
    assert reg["beta"] < 0, f"contrarian slope should be negative, got {reg['beta']:.4f}"
    assert reg["tstat"] < -2.0, f"planted edge should be significant, got t={reg['tstat']:.2f}"


def test_regression_null_insignificant(null_panel):
    """On the null, the slope is statistically indistinguishable from zero."""
    panel, _ = null_panel
    reg = st.predictive_regression(panel)
    assert abs(reg["tstat"]) < 2.5, f"null slope should be insignificant, got t={reg['tstat']:.2f}"
    assert reg["r2"] < 0.05, f"null R-squared should be tiny, got {reg['r2']:.3f}"


def test_regression_deterministic(edge_panel):
    panel, _ = edge_panel
    a = st.predictive_regression(panel)
    b = st.predictive_regression(panel)
    assert a == b


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def test_overlay_columns(null_panel):
    panel, _ = null_panel
    ov = st.timing_overlay(panel)
    assert {"pos", "ret", "gross", "cost", "net", "bh"}.issubset(ov.columns)


def test_overlay_costs_nonnegative(edge_panel):
    panel, _ = edge_panel
    ov = st.timing_overlay(panel, one_way_bps=5.0)
    assert (ov["cost"] >= -1e-12).all(), "costs must be non-negative"
    # net = gross - cost (long/flat, no borrow when not allowed to short)
    np.testing.assert_allclose(ov["net"].values, (ov["gross"] - ov["cost"]).values, atol=1e-12)


def test_overlay_long_flat_positions(null_panel):
    panel, _ = null_panel
    ov = st.timing_overlay(panel, allow_short=False)
    assert set(np.unique(ov["pos"].values)).issubset({0.0, 1.0})


def test_overlay_short_allowed(null_panel):
    panel, _ = null_panel
    ov = st.timing_overlay(panel, allow_short=True)
    assert set(np.unique(ov["pos"].values)).issubset({-1.0, 1.0})
    # short months should carry a borrow cost (>= one-way bps floor)
    short_months = ov["pos"] < 0
    if short_months.any():
        assert (ov.loc[short_months, "cost"] > 0).all()


def test_overlay_edge_beats_null_gross(edge_panel, null_panel):
    """The planted-edge tape should make the contrarian overlay's gross alpha
    (overlay gross minus buy-and-hold) larger than on the null tape."""
    ep, _ = edge_panel
    npn, _ = null_panel
    ov_e = st.timing_overlay(ep, allow_short=True, one_way_bps=0.0, borrow_bps_ann=0.0)
    ov_n = st.timing_overlay(npn, allow_short=True, one_way_bps=0.0, borrow_bps_ann=0.0)
    alpha_e = (ov_e["gross"] - ov_e["bh"]).mean()
    alpha_n = (ov_n["gross"] - ov_n["bh"]).mean()
    assert alpha_e > alpha_n, f"edge alpha {alpha_e:.5f} should exceed null alpha {alpha_n:.5f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regime_summary():
    panel = data.build_real_panel(fetch=False)
    rs = st.regime_summary(panel)
    assert list(rs.index) == ["low", "mid", "high"]
    assert np.isfinite(rs.loc["low", "tstat"])


@requires_cache
def test_real_regression_finite():
    panel = data.build_real_panel(fetch=False)
    reg = st.predictive_regression(panel)
    assert isinstance(reg["tstat"], float) and np.isfinite(reg["tstat"])


@requires_cache
def test_real_contrarian_direction():
    """The real tape should show the contrarian DIRECTION (low regime > high regime),
    even though it is not statistically robust."""
    panel = data.build_real_panel(fetch=False)
    rs = st.regime_summary(panel)
    assert rs.loc["low", "mean_ann"] > rs.loc["high", "mean_ann"]
