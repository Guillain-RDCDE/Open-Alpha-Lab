"""Signals, predictive regression, timing rule, and the study's spine:
the negative froth slope is recovered only when it is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ipo_volume import data, strategy as st  # noqa: E402

REPO_CACHE = data.REPO_CACHE
SHILLER = os.path.join(REPO_CACHE, "shiller_sp500.parquet")


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_froth_signal_burnin_nan(null_panel):
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    assert z.iloc[:4].isna().all(), "First (min_obs-1) years must be NaN"


def test_froth_signal_deterministic(null_panel):
    df, _ = null_panel
    a = st.froth_signal(df, min_obs=5)
    b = st.froth_signal(df, min_obs=5)
    pd.testing.assert_series_equal(a, b)


def test_froth_signal_is_past_only(live_panel):
    """The z-score at year i must depend only on counts up to i (expanding window).

    Mutating a future count must not change an earlier z-score.
    """
    df, _ = live_panel
    z_full = st.froth_signal(df, min_obs=5)
    df2 = df.copy()
    df2.loc[df2.index[-1], "ipo_count"] = 99999  # blow up the last year
    z_mut = st.froth_signal(df2, min_obs=5)
    # All but the last value must be identical (future leak would change earlier z).
    pd.testing.assert_series_equal(z_full.iloc[:-1], z_mut.iloc[:-1])


def test_froth_signal_standardized_scale(live_panel):
    df, _ = live_panel
    z = st.froth_signal(df, min_obs=5).dropna()
    # Expanding z-scores are roughly standardized; sanity bound.
    assert z.abs().max() < 8.0


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_regression_keys(null_panel):
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    assert set(["alpha", "beta", "beta_t", "r2", "n", "corr"]).issubset(reg)


def test_regression_short_series_safe():
    z = pd.Series([0.1, 0.2])
    y = pd.Series([0.01, -0.02])
    reg = st.predictive_regression(z, y)
    assert reg["n"] == 2
    assert np.isnan(reg["beta"]) or isinstance(reg["beta"], float)


def test_null_panel_slope_insignificant(null_panel):
    """beta=0 planted: regression slope must not clear the inference bar."""
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    assert np.isfinite(reg["beta_t"])
    assert abs(reg["beta_t"]) < 2.5, f"Null |t| too large: {reg['beta_t']:.2f}"


def test_live_panel_recovers_negative_slope(live_panel):
    """beta=-0.06 planted: regression recovers a significantly negative slope."""
    df, _ = live_panel
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    assert reg["beta"] < 0.0, f"Expected negative slope, got {reg['beta']:.4f}"
    assert reg["beta_t"] < -2.0, f"Expected t < -2, got {reg['beta_t']:.2f}"


# ---------------------------------------------------------------------------
# Timing rule
# ---------------------------------------------------------------------------
def test_timing_columns(null_panel):
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    tm = st.timing_returns(z, df["fwd_return"])
    assert set(["position", "strat_gross", "strat_net", "buy_hold"]).issubset(tm.columns)


def test_timing_positions_binary(null_panel):
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    tm = st.timing_returns(z, df["fwd_return"])
    assert set(tm["position"].unique()).issubset({0.0, 1.0})


def test_timing_costs_reduce_returns(live_panel):
    """Net must be <= gross (costs are non-negative)."""
    df, _ = live_panel
    z = st.froth_signal(df, min_obs=5)
    tm = st.timing_returns(z, df["fwd_return"], one_way_bps=10.0)
    assert (tm["strat_net"] <= tm["strat_gross"] + 1e-12).all()


def test_timing_long_only_no_short(null_panel):
    """The rule is long-only: positions never go negative (no borrow, no short)."""
    df, _ = null_panel
    z = st.froth_signal(df, min_obs=5)
    tm = st.timing_returns(z, df["fwd_return"])
    assert (tm["position"] >= 0.0).all()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.1)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert s["n"] == 2


def test_annualise_identity():
    s = st.summarize(pd.Series([0.05] * 30))
    out = st.annualise(s, periods=1)
    assert abs(out["mean_ann"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Real-data tests: guarded by Shiller cache presence
# ---------------------------------------------------------------------------
requires_shiller = pytest.mark.skipif(
    not os.path.exists(SHILLER),
    reason="shiller_sp500.parquet repo cache absent; covered by synthetic tests",
)


@requires_shiller
def test_real_regression_runs():
    df = data.fetch_sp500_annual()
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    assert np.isfinite(reg["beta_t"]), "Real-tape regression t must be finite"
    assert reg["n"] > 30


@requires_shiller
def test_real_signal_is_a_mirage():
    """The headline: on the real tape, the full-sample froth slope does NOT clear t=2.

    This is the study's verdict baked into a regression test -- if a data revision
    suddenly made IPO volume a real predictor, this assertion would flag it.
    """
    df = data.fetch_sp500_annual()
    z = st.froth_signal(df, min_obs=5)
    reg = st.predictive_regression(z, df["fwd_return"])
    assert abs(reg["beta_t"]) < 2.0, (
        f"Full-sample froth slope unexpectedly significant: t={reg['beta_t']:.2f}"
    )


@requires_shiller
def test_real_timing_underperforms_buy_hold():
    """The tradable wrapper sits out good years and loses to buy-and-hold."""
    df = data.fetch_sp500_annual()
    z = st.froth_signal(df, min_obs=5)
    tm = st.timing_returns(z, df["fwd_return"], one_way_bps=5.0)
    assert tm["strat_net"].mean() < tm["buy_hold"].mean(), (
        "Froth timing rule should not beat buy-and-hold on the real tape"
    )
