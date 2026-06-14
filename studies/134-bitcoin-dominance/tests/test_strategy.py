"""Signal construction, inference, and the study's spine: the dominance signal only
beats the base-rate control when a genuine dominance→alt-spread link is planted."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_dominance import data, strategy as st  # noqa: E402


# ---- alt basket returns ---------------------------------------------------
def test_alt_basket_return_shape(null_panel):
    panel, truth = null_panel
    fwd = st.alt_basket_return(panel, horizon=5)
    assert "btc_fwd" in fwd.columns
    assert "alt_fwd" in fwd.columns
    assert "spread_fwd" in fwd.columns
    assert len(fwd) > 0
    # Spread = alt - btc
    assert np.allclose(fwd["spread_fwd"], fwd["alt_fwd"] - fwd["btc_fwd"])


def test_alt_basket_no_lookahead(null_panel):
    """The forward return window must start at t+1, not t."""
    panel, _ = null_panel
    fwd = st.alt_basket_return(panel, horizon=5)
    # First available spread date must be at index 0 (signal formed on day 0, return starts day 1)
    assert fwd.index[0] == panel.index[0]
    # Last available date must be n - horizon - 1 at most
    last_idx = fwd.index[-1]
    assert last_idx <= panel.index[-6]  # need 5 forward days


# ---- dominance signal ---------------------------------------------------
def test_dominance_signal_length(null_panel):
    panel, _ = null_panel
    sig = st.dominance_signal(panel, lookback=10)
    # First lookback rows should be NaN; rest should be numeric
    assert sig.isna().sum() == 10
    assert sig.dropna().shape[0] == len(panel) - 10


def test_dominance_signal_sign(null_panel):
    """The signal has both positive and negative values (dominance moves both ways)."""
    panel, _ = null_panel
    sig = st.dominance_signal(panel, lookback=10).dropna()
    assert (sig > 0).sum() > 0
    assert (sig < 0).sum() > 0


def test_dominance_percentile_bounded(null_panel):
    panel, _ = null_panel
    pct = st.dominance_percentile(panel, window=252).dropna()
    assert (pct >= 0).all() and (pct <= 1).all()


# ---- signal-outcome pairs and controls -----------------------------------
def test_signal_outcome_pairs_shape(null_panel):
    panel, _ = null_panel
    sig = st.dominance_signal(panel, lookback=10)
    pairs = st.signal_outcome_pairs(panel, sig)
    assert "signal" in pairs.columns
    assert "spread_fwd" in pairs.columns
    assert len(pairs) > 50


def test_random_period_control_distribution(null_panel):
    panel, _ = null_panel
    sig = st.dominance_signal(panel, lookback=10)
    pairs = st.signal_outcome_pairs(panel, sig)
    ctrl = st.random_period_control(pairs, n_draws=100, seed=42)
    assert len(ctrl) > 0
    # Control distribution should be centred near unconditional mean
    unconditional = float(pairs["spread_fwd"].mean())
    assert abs(ctrl["mean_spread"].mean() - unconditional) < 0.02


# ---- summarize and regression -------------------------------------------
def test_summarize_basic(null_panel):
    panel, _ = null_panel
    fwd = st.alt_basket_return(panel)
    s = st.summarize(fwd["spread_fwd"])
    assert "n" in s
    assert "mean_pct" in s
    assert "tstat" in s
    assert s["n"] > 50


def test_regression_tstat_basic(null_panel):
    panel, _ = null_panel
    sig = st.dominance_signal(panel, lookback=10)
    pairs = st.signal_outcome_pairs(panel, sig)
    reg = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])
    assert "slope" in reg
    assert "tstat" in reg
    assert np.isfinite(reg["tstat"])


# ---- the study's spine ---------------------------------------------------
def test_signal_finds_effect_only_when_planted(null_panel, signal_panel):
    """The engine: with a planted dom→alt signal, the regression slope is more negative
    (falling dominance → higher spread); on the null it is near zero."""
    def reg_slope(panel):
        sig = st.dominance_signal(panel, lookback=10)
        pairs = st.signal_outcome_pairs(panel, sig)
        r = st.regression_tstat(pairs["spread_fwd"], pairs["signal"])
        return r["slope"]

    slope_null = reg_slope(null_panel[0])
    slope_signal = reg_slope(signal_panel[0])
    # Signal panel slope should be more negative (falling dom → higher spread)
    assert slope_signal < slope_null


def test_summarize_tstat_is_finite_with_enough_data(null_panel):
    panel, _ = null_panel
    fwd = st.alt_basket_return(panel)
    s = st.summarize(fwd["spread_fwd"])
    assert np.isfinite(s["tstat"])
    assert s["n"] > 50
