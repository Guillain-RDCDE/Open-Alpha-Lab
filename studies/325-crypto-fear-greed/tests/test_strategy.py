"""Strategy tests: forward returns (one-day lag), regime binning, contrarian spread,
costs, controls, and the study's spine -- the fear-minus-greed overlay only clears t>2
when genuine mean reversion is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_fear_greed import data, strategy as st  # noqa: E402


def _panel(df):
    return data.panel_from_close(df["close"])


# ---------------------------------------------------------------------------
# Forward returns -- the one-day execution lag
# ---------------------------------------------------------------------------
def test_forward_returns_columns(null_tape):
    df, _ = null_tape
    fr = st.forward_returns(_panel(df))
    assert list(fr.columns) == ["fng", "fwd_ret"]


def test_forward_returns_no_lookahead(null_tape):
    df, _ = null_tape
    panel = _panel(df)
    fr = st.forward_returns(panel)
    close = panel["close"]
    expected = close.iloc[1] / close.iloc[0] - 1.0
    assert abs(fr["fwd_ret"].iloc[0] - expected) < 1e-12


def test_forward_returns_drops_last_row(null_tape):
    df, _ = null_tape
    panel = _panel(df)
    fr = st.forward_returns(panel)
    assert len(fr) == len(panel) - 1


# ---------------------------------------------------------------------------
# Regime binning
# ---------------------------------------------------------------------------
def test_regime_band_known_values():
    s = pd.Series([5.0, 30.0, 50.0, 65.0, 90.0])
    band = st.regime_band(s)
    assert list(band) == ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def test_regime_means_covers_bands(control_tape):
    df, _ = control_tape
    rm = st.regime_means(st.forward_returns(_panel(df)))
    assert set(rm.index) == set(st.BAND_LABELS)
    assert "tstat" in rm.columns and "mean_ann" in rm.columns


# ---------------------------------------------------------------------------
# Contrarian position and spread
# ---------------------------------------------------------------------------
def test_contrarian_position_signs():
    fng = pd.Series([10.0, 50.0, 80.0])
    pos = st.contrarian_position(fng)
    assert list(pos) == [1.0, 0.0, -1.0]


def test_spread_equals_pos_times_fwd(null_tape):
    df, _ = null_tape
    fr = st.forward_returns(_panel(df))
    pos = st.contrarian_position(fr["fng"])
    sp = st.spread_returns(fr)
    np.testing.assert_allclose(sp.values, (pos * fr["fwd_ret"]).values, atol=1e-12)


def test_turnover_states_sum_to_one(null_tape):
    df, _ = null_tape
    ts = st.turnover_stats(st.forward_returns(_panel(df)))
    assert abs(ts["pct_long"] + ts["pct_flat"] + ts["pct_short"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Costs: net <= gross
# ---------------------------------------------------------------------------
def test_net_spread_below_gross(control_tape):
    df, _ = control_tape
    fr = st.forward_returns(_panel(df))
    gross = st.spread_returns(fr).mean()
    net = st.net_spread_returns(fr, one_way_bps=10.0, borrow_bps_ann=1000.0).mean()
    assert net <= gross + 1e-12, "Net spread must not exceed gross (costs are a drag)"


# ---------------------------------------------------------------------------
# Random-timing null
# ---------------------------------------------------------------------------
def test_random_timing_null_reproducible(null_tape):
    df, _ = null_tape
    fr = st.forward_returns(_panel(df))
    a = st.random_timing_null(fr, n_draws=200, seed=1)
    b = st.random_timing_null(fr, n_draws=200, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_timing_null_centered(null_tape):
    df, _ = null_tape
    fr = st.forward_returns(_panel(df))
    null = st.random_timing_null(fr, n_draws=400, seed=42)
    assert abs(null.mean()) < 0.005, f"Null spread mean too far from 0: {null.mean():.5f}"


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def test_block_bootstrap_ci_orders(control_tape):
    df, _ = control_tape
    fr = st.forward_returns(_panel(df))
    ci = st.block_bootstrap_sharpe_ci(st.spread_returns(fr), n_boot=300, seed=1)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]
    assert 0.0 <= ci["frac_negative"] <= 1.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(200) * 0.03)
    s = st.summarize(r)
    assert {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}.issubset(s)


def test_annualise():
    s = st.summarize(pd.Series([0.001] * 400))
    out = st.annualise(s, periods=365)
    assert abs(out["mean_ann"] - 365 * 0.001) < 1e-9


# ---------------------------------------------------------------------------
# The spine: the reversion knob switches significance
# ---------------------------------------------------------------------------
def test_reversion_knob_makes_spread_significant(control_tape):
    """With strong planted mean reversion, fear-minus-greed clears t>2."""
    df, _ = control_tape
    fr = st.forward_returns(_panel(df))
    s = st.summarize(st.spread_returns(fr))
    assert s["mean"] > 0.0, f"Planted reversion should give positive spread, got {s['mean']:.5f}"
    assert s["tstat"] > 2.0, f"Planted spread should clear t>2, got {s['tstat']:.2f}"


def test_null_spread_not_significant(null_tape):
    """On the pure random walk the spread is finite and not wildly significant."""
    df, _ = null_tape
    fr = st.forward_returns(_panel(df))
    s = st.summarize(st.spread_returns(fr))
    assert np.isfinite(s["mean"]) and np.isfinite(s["tstat"])
    assert abs(s["tstat"]) < 2.0, f"Null |t| should stay within band, got {s['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
_LOCAL_BTC = data._local_cache_path()
requires_cache = pytest.mark.skipif(
    not (os.path.exists(data.BTC_CACHE) or os.path.exists(_LOCAL_BTC)),
    reason="BTC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_spread_is_finite_float():
    panel = data.real_panel(fetch=False)
    fr = st.forward_returns(panel)
    s = st.summarize(st.spread_returns(fr))
    assert isinstance(s["tstat"], float) and np.isfinite(s["tstat"])


@requires_cache
def test_real_spread_does_not_clear_bar():
    """Documented headline: the real contrarian spread does NOT clear t>2 (it's negative)."""
    panel = data.real_panel(fetch=False)
    fr = st.forward_returns(panel)
    s = st.summarize(st.spread_returns(fr))
    assert abs(s["tstat"]) < 2.0, f"Headline is a mirage; real |t| should be <2, got {s['tstat']:.2f}"
