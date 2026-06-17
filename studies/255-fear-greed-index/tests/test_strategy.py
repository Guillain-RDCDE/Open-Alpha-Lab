"""Signals, regime binning, contrarian spread, costs, controls, and the study's spine:
the fear-minus-greed overlay only clears t>2 when a contrarian premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fear_greed_index import data, strategy as st  # noqa: E402

GSPC_CACHE = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Forward returns -- the one-week execution lag
# ---------------------------------------------------------------------------
def test_forward_returns_columns(null_panel):
    df, _ = null_panel
    fr = st.forward_returns(df)
    assert list(fr.columns) == ["fng", "fwd_ret"]
    assert len(fr) == len(df) - 1, "Last row drops (no forward return)"


def test_forward_returns_no_lookahead(null_panel):
    """fwd_ret at row t must equal close[t+1]/close[t]-1 (uses future, not present)."""
    df, _ = null_panel
    fr = st.forward_returns(df)
    close = df["close"]
    expected = close.iloc[1] / close.iloc[0] - 1.0
    assert abs(fr["fwd_ret"].iloc[0] - expected) < 1e-12


def test_forward_returns_deterministic(null_panel):
    df, _ = null_panel
    a = st.forward_returns(df)
    b = st.forward_returns(df)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Regime binning
# ---------------------------------------------------------------------------
def test_regime_band_labels(null_panel):
    df, _ = null_panel
    band = st.regime_band(df["fng"])
    assert set(band.dropna().unique()).issubset(set(st.BAND_LABELS))


def test_regime_band_known_values():
    s = pd.Series([5.0, 30.0, 50.0, 65.0, 90.0])
    band = st.regime_band(s)
    assert list(band) == ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def test_regime_means_covers_bands(live_panel):
    df, _ = live_panel
    fr = st.forward_returns(df)
    rm = st.regime_means(fr)
    assert set(rm.index) == set(st.BAND_LABELS)
    assert "tstat" in rm.columns and "mean_ann" in rm.columns


# ---------------------------------------------------------------------------
# Contrarian position and spread
# ---------------------------------------------------------------------------
def test_contrarian_position_signs():
    fng = pd.Series([10.0, 50.0, 80.0])
    pos = st.contrarian_position(fng)
    assert list(pos) == [1.0, 0.0, -1.0]


def test_spread_equals_pos_times_fwd(null_panel):
    df, _ = null_panel
    fr = st.forward_returns(df)
    pos = st.contrarian_position(fr["fng"])
    sp = st.spread_returns(fr)
    np.testing.assert_allclose(sp.values, (pos * fr["fwd_ret"]).values, atol=1e-12)


def test_turnover_states_sum_to_one(null_panel):
    df, _ = null_panel
    fr = st.forward_returns(df)
    ts = st.turnover_stats(fr)
    assert abs(ts["pct_long"] + ts["pct_flat"] + ts["pct_short"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Costs: net spread <= gross spread mean
# ---------------------------------------------------------------------------
def test_net_spread_below_gross(live_panel):
    df, _ = live_panel
    fr = st.forward_returns(df)
    gross = st.spread_returns(fr).mean()
    net = st.net_spread_returns(fr, one_way_bps=5.0, borrow_bps_ann=50.0).mean()
    assert net <= gross + 1e-12, "Net spread must not exceed gross (costs are a drag)"


# ---------------------------------------------------------------------------
# Random-timing null
# ---------------------------------------------------------------------------
def test_random_timing_null_reproducible(null_panel):
    df, _ = null_panel
    fr = st.forward_returns(df)
    a = st.random_timing_null(fr, n_draws=200, seed=1)
    b = st.random_timing_null(fr, n_draws=200, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_timing_null_centered(null_panel):
    """Under the null the shuffled-sentiment spread mean is small (near zero)."""
    df, _ = null_panel
    fr = st.forward_returns(df)
    null = st.random_timing_null(fr, n_draws=500, seed=42)
    assert abs(null.mean()) < 0.003, f"Null spread mean too far from 0: {null.mean():.5f}"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(60) * 0.02)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_annualise_weekly():
    s = st.summarize(pd.Series([0.001] * 80))
    out = st.annualise_weekly(s, periods=52)
    assert abs(out["mean_ann"] - 52 * 0.001) < 1e-10


# ---------------------------------------------------------------------------
# The spine: the contrarian premium knob switches significance
# ---------------------------------------------------------------------------
def test_premium_knob_makes_spread_significant(live_panel):
    """With a strong planted contrarian premium, fear-minus-greed clears t>2."""
    df, _ = live_panel
    fr = st.forward_returns(df)
    s = st.summarize(st.spread_returns(fr))
    assert s["mean"] > 0.0, f"Planted contrarian premium should give positive spread, got {s['mean']:.5f}"
    assert s["tstat"] > 2.0, f"Planted spread should be significant (t>2), got {s['tstat']:.2f}"


def test_null_spread_not_significant(null_panel):
    """On the null panel the spread is finite and not wildly significant."""
    df, _ = null_panel
    fr = st.forward_returns(df)
    s = st.summarize(st.spread_returns(fr))
    assert np.isfinite(s["mean"]) and np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 4.0, f"Null |t| too large: {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_spread_is_finite_float():
    panel = data.real_panel(fetch=False)
    fr = st.forward_returns(panel)
    s = st.summarize(st.spread_returns(fr))
    assert isinstance(s["tstat"], float) and np.isfinite(s["tstat"])


@requires_cache
def test_real_spread_is_weak():
    """Documented headline: the real contrarian spread does NOT clear the t>2 bar."""
    panel = data.real_panel(fetch=False)
    fr = st.forward_returns(panel)
    s = st.summarize(st.spread_returns(fr))
    assert abs(s["tstat"]) < 2.0, f"Headline claims a mirage; real |t| should be <2, got {s['tstat']:.2f}"
