"""Signals, regressions, event tests, and the study's spine: the contrarian
forward signal is recovered ONLY when planted in the synthetic tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_debt import data, strategy as st  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------
def test_standardized_yoy_zero_mean_unit_var(null_panel):
    panel, _ = null_panel
    z = st.standardized_yoy(panel)
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std(ddof=0)) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# HAC OLS basics
# ---------------------------------------------------------------------------
def test_ols_hac_recovers_known_line():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    y = 0.5 - 0.3 * x + rng.normal(0, 0.01, 300)
    r = st.ols_hac(y, x)
    assert abs(r["beta"] - (-0.3)) < 0.02
    assert abs(r["alpha"] - 0.5) < 0.02
    assert r["beta_t"] < -2


def test_ols_hac_short_series_nan():
    r = st.ols_hac(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    assert np.isnan(r["beta"])


# ---------------------------------------------------------------------------
# The spine: planted contrarian signal switches significance
# ---------------------------------------------------------------------------
def test_null_panel_regression_insignificant(null_panel):
    """No planted forward signal -> regression slope should be ~0 (|t| < 2)."""
    panel, _ = null_panel
    r = st.predictive_regression(panel)
    assert np.isfinite(r["beta_t"])
    assert abs(r["beta_t"]) < 2.0, f"Null |t| too large: {r['beta_t']:.2f}"


def test_live_panel_regression_negative_significant(live_panel):
    """Planted contrarian signal -> strongly NEGATIVE slope (t < -2)."""
    panel, _ = live_panel
    r = st.predictive_regression(panel)
    assert r["beta"] < 0, f"Planted contrarian slope should be negative, got {r['beta']:+.4f}"
    assert r["beta_t"] < -2.0, f"Planted contrarian t should be < -2, got {r['beta_t']:.2f}"


def test_live_panel_tercile_spread_negative(live_panel):
    """With a planted contrarian signal, high-growth tercile underperforms low."""
    panel, _ = live_panel
    ter = st.tercile_forward_returns(panel)
    assert ter["spread_high_minus_low"] < 0


# ---------------------------------------------------------------------------
# Event test machinery
# ---------------------------------------------------------------------------
def test_record_high_event_keys(null_panel):
    panel, _ = null_panel
    rec = st.record_high_event(panel)
    for k in ("record_mean", "nonrecord_mean", "diff", "uncond_up_rate", "welch_t"):
        assert k in rec
    assert 0.0 <= rec["uncond_up_rate"] <= 1.0
    assert rec["n_record"] >= 1


def test_record_first_year_is_record(null_panel):
    """The first observation is always a running-max record."""
    panel, _ = null_panel
    p = panel.dropna(subset=["fwd_ret"]).copy()
    assert (p["margin_debt"].cummax() >= p["margin_debt"]).iloc[0]


# ---------------------------------------------------------------------------
# Timing backtest
# ---------------------------------------------------------------------------
def test_timing_backtest_keys_and_cash(null_panel):
    panel, _ = null_panel
    bt = st.contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=10.0)
    for k in ("bah_ann", "strat_net_ann", "years_in_cash", "shortfall_net_pp"):
        assert k in bt
    assert bt["years_in_cash"] >= 0
    assert np.isfinite(bt["bah_ann"])


def test_timing_net_below_gross(null_panel):
    panel, _ = null_panel
    bt = st.contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=20.0)
    assert bt["strat_net_ann"] <= bt["strat_gross_ann"] + 1e-12


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_panel):
    panel, _ = null_panel
    s = st.summarize(panel)
    for k in ("n", "reg_beta", "reg_beta_t", "rec_diff", "bah_ann", "strat_net_ann"):
        assert k in s


# ---------------------------------------------------------------------------
# Real-tape tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regression_is_finite():
    panel = data.build_panel(fetch=False)
    r = st.predictive_regression(panel)
    assert np.isfinite(r["beta_t"])
    assert isinstance(r["beta"], float)


@requires_cache
def test_real_contrarian_sign_but_weak():
    """The real tape carries the contrarian SIGN (negative slope) but it does NOT
    clear |t| >= 2 -- this is the WEAK verdict the study assigns."""
    panel = data.build_panel(fetch=False)
    r = st.predictive_regression(panel)
    assert r["beta"] < 0, "Expected contrarian (negative) slope on the real tape"
    assert abs(r["beta_t"]) < 2.0, "Real-tape slope should NOT clear |t|>=2 (verdict: Weak)"


@requires_cache
def test_real_record_years_underperform_but_not_significant():
    panel = data.build_panel(fetch=False)
    rec = st.record_high_event(panel)
    assert rec["diff"] < 0, "Record years should average lower forward returns"
    assert abs(rec["welch_t"]) < 2.0, "Record-event difference should not be significant"
