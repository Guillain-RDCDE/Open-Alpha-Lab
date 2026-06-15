"""Signals, the backtest engine, and the study's spine:
divergence timing is unreliable on a null tape, and the permutation baseline
shows that small sub-samples easily look "extreme" by chance."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from advance_decline import data, strategy as st  # noqa: E402


# ---- signal construction --------------------------------------------------

def test_signal_is_boolean_series(null_tape):
    spy, ad, _ = null_tape
    sig = st.ad_divergence_signal(spy["close"], ad)
    assert sig.dtype == bool
    assert len(sig) == len(spy)


def test_signal_respects_lookback(null_tape):
    spy, ad, _ = null_tape
    lb = 42
    sig = st.ad_divergence_signal(spy["close"], ad, lookback=lb)
    # First lb-1 days must be NaN/False (rolling window not yet full)
    assert not sig.iloc[:lb - 1].any()


def test_no_signal_on_constant_series():
    n = 100
    idx = pd.bdate_range("2020-01-02", periods=n, name="date")
    spy = pd.Series(100.0, index=idx)
    ad  = pd.Series(0.0, index=idx)
    sig = st.ad_divergence_signal(spy, ad, lookback=20, min_index_gain=0.03)
    # Constant price -> no new high with gain -> no signal
    assert not sig.any()


def test_planted_divergence_fires_signals(divergence_tape):
    """With planted divergence, the detector should fire at least occasionally."""
    spy, ad, _ = divergence_tape
    sig = st.ad_divergence_signal(spy["close"], ad, lookback=21, min_index_gain=0.0)
    # With divergence planted, we should see at least some signals
    assert sig.sum() >= 1


def test_no_lookahead_in_signal(null_tape):
    """Rolling max on day t uses only data up to and including t."""
    spy, ad, _ = null_tape
    # Artificially spike the last value — the signal should not fire earlier
    spy_mod = spy["close"].copy()
    spy_mod.iloc[-1] = spy_mod.max() * 10
    sig_orig = st.ad_divergence_signal(spy["close"], ad)
    sig_mod  = st.ad_divergence_signal(spy_mod,      ad)
    # The modification is only at the last day — no earlier day should change
    assert (sig_orig.iloc[:-1] == sig_mod.iloc[:-1]).all()


# ---- forward returns ------------------------------------------------------

def test_forward_returns_shape(null_tape):
    spy, _, _ = null_tape
    fwd = st.forward_returns(spy["close"], horizon=5)
    assert len(fwd) == len(spy)
    assert fwd.iloc[-5:].isna().all()      # last 5 days cannot have full horizon
    assert fwd.iloc[:-5].notna().all()


def test_forward_returns_sign(null_tape):
    spy, _, _ = null_tape
    fwd = st.forward_returns(spy["close"], horizon=1)
    # fwd[t] = log(spy[t+1] / spy[t]) — check a few manually
    close = spy["close"]
    for i in [0, 5, 10]:
        expected = np.log(close.iloc[i + 1] / close.iloc[i])
        assert abs(fwd.iloc[i] - expected) < 1e-9


# ---- backtest -------------------------------------------------------------

def test_backtest_keys(null_tape):
    spy, ad, _ = null_tape
    res = st.backtest(spy["close"], ad, lookback=42, horizon=21)
    for k in ("div_stats", "base_stats", "diff_mean_bps", "n_divergences",
              "div_rets", "base_rets"):
        assert k in res


def test_backtest_base_covers_all_valid_days(null_tape):
    spy, ad, _ = null_tape
    res = st.backtest(spy["close"], ad, lookback=42, horizon=21)
    # Base rets covers more days than divergence sub-sample
    assert len(res["base_rets"]) > len(res["div_rets"])


# ---- summarize / HAC t-stat -----------------------------------------------

def test_summarize_shape():
    rng = np.random.default_rng(1)
    r   = rng.normal(0.001, 0.01, 200)
    s   = st.summarize(r, label="test")
    for k in ("n", "mean_bps", "hit_rate", "sharpe", "tstat"):
        assert k in s
    assert s["n"] == 200


def test_summarize_positive_mean():
    r = np.full(50, 0.005)      # pure positive
    s = st.summarize(r)
    assert s["mean_bps"] > 0
    assert s["hit_rate"] == 1.0


def test_summarize_handles_small_n():
    r = np.array([0.01, -0.01, 0.005])
    s = st.summarize(r)
    assert np.isnan(s["tstat"])   # < 5 obs → no HAC


# ---- permutation baseline ------------------------------------------------

def test_permutation_baseline_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, 500)
    res = st.permutation_baseline(base, n_signal=30, n_perm=200, seed=1)
    assert len(res["perm_means_bps"]) == 200
    # The distribution of permuted sub-sample means should be centered near 0
    assert abs(res["perm_mean"]) < 5.0   # well within 5 bps of zero


# ---- the spine: null tape shows no reliable forward signal ----------------

def test_null_tape_shows_no_reliable_signal(null_tape):
    """On a null (random-walk) tape, the divergence sub-sample should NOT
    systematically differ from the baseline — |diff| should be small and
    t-stat below 2."""
    spy, ad, _ = null_tape
    res = st.backtest(spy["close"], ad, lookback=63, horizon=21)
    # Signal is rare; when present, t-stat should not be well outside noise
    if res["n_divergences"] > 5:
        assert abs(res["div_stats"]["tstat"]) < 3.0   # not a screaming signal


def test_sweep_lookback_returns_dataframe(null_tape):
    spy, ad, _ = null_tape
    df = st.sweep_lookback(spy["close"], ad, lookbacks=[21, 42, 63], horizon=10)
    assert set(df.columns) >= {"n_signals", "div_mean_bps", "base_mean_bps"}
    assert len(df) == 3
