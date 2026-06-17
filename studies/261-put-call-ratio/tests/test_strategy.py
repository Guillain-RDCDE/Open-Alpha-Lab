"""Signals, timing returns, predictive regression, and the study's spine:
the contrarian rule only shows an edge when a contrarian effect is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from put_call_ratio import data, strategy as st  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Signal construction -- no look-ahead
# ---------------------------------------------------------------------------
def test_extreme_signal_burnin(null_panel):
    df, _ = null_panel
    sig = st.extreme_signal(df["pcr"], q=0.80, min_history=36)
    assert (sig.iloc[:36] == 0).all(), "Signal must be flat during burn-in"


def test_extreme_signal_binary(null_panel):
    df, _ = null_panel
    sig = st.extreme_signal(df["pcr"], q=0.80, min_history=36)
    assert set(np.unique(sig.values)).issubset({0, 1})


def test_extreme_signal_deterministic(null_panel):
    df, _ = null_panel
    a = st.extreme_signal(df["pcr"], q=0.80)
    b = st.extreme_signal(df["pcr"], q=0.80)
    pd.testing.assert_series_equal(a, b)


def test_zscore_signal_burnin(null_panel):
    df, _ = null_panel
    z = st.zscore_signal(df["pcr"], min_history=36)
    assert z.iloc[:36].isna().all()
    assert z.iloc[36:].notna().all()


def test_extreme_signal_fraction_reasonable(null_panel):
    """With q=0.80 the in-market fraction (post-burn-in) should be roughly 20%."""
    df, _ = null_panel
    sig = st.extreme_signal(df["pcr"], q=0.80, min_history=36).iloc[36:]
    frac = sig.mean()
    assert 0.10 < frac < 0.35, f"Expected ~20% extreme-fear flags, got {frac:.2%}"


# ---------------------------------------------------------------------------
# Timing returns
# ---------------------------------------------------------------------------
def test_timing_returns_columns(null_panel):
    df, _ = null_panel
    tr = st.timing_returns(df, q=0.80)
    assert {"signal", "bh", "strat", "strat_net", "in_mkt"}.issubset(tr.columns)


def test_timing_strat_is_signal_times_bh(null_panel):
    df, _ = null_panel
    tr = st.timing_returns(df, q=0.80)
    np.testing.assert_allclose(
        tr["strat"].values, tr["signal"].values * tr["bh"].values, atol=1e-12
    )


def test_timing_net_le_gross_when_switching(null_panel):
    """Net return can only be <= gross (costs are non-negative)."""
    df, _ = null_panel
    tr = st.timing_returns(df, q=0.80, one_way_bps=5.0)
    assert (tr["strat_net"] <= tr["strat"] + 1e-12).all()


# ---------------------------------------------------------------------------
# Random-timing null
# ---------------------------------------------------------------------------
def test_random_timing_reproducible(null_panel):
    df, _ = null_panel
    a = st.random_timing_returns(df, in_mkt_frac=0.2, n_draws=100, seed=1)
    b = st.random_timing_returns(df, in_mkt_frac=0.2, n_draws=100, seed=1)
    pd.testing.assert_series_equal(a, b)


def test_random_timing_centred(null_panel):
    """Random in-market mean should centre near the full-sample mean return."""
    df, _ = null_panel
    full_mean = df["fwd_ret"].iloc[36:].mean()
    rand = st.random_timing_returns(df, in_mkt_frac=0.2, n_draws=500, seed=42)
    assert abs(rand.mean() - full_mean) < 0.01


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_regression_keys(null_panel):
    df, _ = null_panel
    reg = st.predictive_regression(df)
    assert set(reg) == {"slope", "tstat", "r2", "n"}


def test_regression_null_insignificant(null_panel):
    """Under the null the slope t-stat must be small."""
    df, _ = null_panel
    reg = st.predictive_regression(df)
    assert abs(reg["tstat"]) < 3.0, f"Null slope t too large: {reg['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.04)
    s = st.summarize(r)
    assert {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}.issubset(s)


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 0.12) < 1e-10


# ---------------------------------------------------------------------------
# The spine: planted contrarian beta switches the verdict
# ---------------------------------------------------------------------------
def test_planted_beta_recovered(live_panel):
    """With a planted positive contrarian beta, the regression slope is positive & significant."""
    df, _ = live_panel
    reg = st.predictive_regression(df)
    assert reg["slope"] > 0, f"Expected positive contrarian slope, got {reg['slope']:.4f}"
    assert reg["tstat"] > 2.0, f"Planted effect should be significant (t>2), got {reg['tstat']:.2f}"


def test_null_beta_no_edge(null_panel):
    """Under the null the contrarian slope is not significant."""
    df, _ = null_panel
    reg = st.predictive_regression(df)
    assert abs(reg["tstat"]) < 2.5


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regression_finite():
    df = data.build_real_panel(fetch=False)
    reg = st.predictive_regression(df)
    assert np.isfinite(reg["tstat"])


@requires_cache
def test_real_timing_runs():
    df = data.build_real_panel(fetch=False)
    tr = st.timing_returns(df, q=0.80)
    assert len(tr) > 50
    s = st.summarize(tr["strat"])
    assert np.isfinite(s["tstat"])
