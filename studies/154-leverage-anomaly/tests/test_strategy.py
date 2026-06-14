"""The quintile-spread engine, the random control, and the study's spine:
the low-minus-high spread beats random only when a premium is planted."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage_anomaly import strategy as st  # noqa: E402


# ---- quintile_spread output contract ----------------------------------------
def test_spread_columns(anomaly_panel):
    lev, fwd, _ = anomaly_panel
    out = st.quintile_spread(lev, fwd)
    assert set(["low", "high", "market", "spread", "n", "n_low", "n_high"]).issubset(out.columns)
    assert len(out) == len(lev)


def test_spread_is_low_minus_high_by_default(anomaly_panel):
    lev, fwd, _ = anomaly_panel
    out = st.quintile_spread(lev, fwd, low_is_good=True)
    assert np.allclose(out["spread"], out["low"] - out["high"])


def test_spread_inverts_with_low_is_good_false(anomaly_panel):
    lev, fwd, _ = anomaly_panel
    a = st.quintile_spread(lev, fwd, low_is_good=True)["spread"]
    b = st.quintile_spread(lev, fwd, low_is_good=False)["spread"]
    assert np.allclose(a.values, -b.values)


def test_quintile_sizes_symmetric(null_panel):
    lev, fwd, _ = null_panel
    out = st.quintile_spread(lev, fwd, q=0.20)
    # Each quintile should hold ~20% of the universe
    for _, row in out.iterrows():
        assert row["n_low"] > 0
        assert row["n_high"] > 0
        assert abs(row["n_low"] - row["n_high"]) <= 3   # nearly symmetric


def test_spread_needs_minimum_universe(null_panel):
    """Drop years where the aligned universe falls below 20: no row emitted."""
    lev, fwd, _ = null_panel
    # Use a tiny panel that won't hit the 20-firm minimum
    tiny_lev = lev.iloc[:, :10]
    out = st.quintile_spread(tiny_lev, fwd)
    assert out.empty


# ---- random control ----------------------------------------------------------
def test_random_excess_is_mean_zero_on_null(null_panel):
    """Under the null the random-portfolio excess should be centred near zero."""
    lev, fwd, _ = null_panel
    exc = st.random_portfolio_excess(lev, fwd, n_draws=200, seed=1)
    assert abs(exc.mean()) < 0.05   # within 5 pp of zero
    assert len(exc) > 0


def test_random_excess_reproducible(null_panel):
    lev, fwd, _ = null_panel
    a = st.random_portfolio_excess(lev, fwd, n_draws=50, seed=42)
    b = st.random_portfolio_excess(lev, fwd, n_draws=50, seed=42)
    assert np.allclose(a.values, b.values)


# ---- summarize --------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series([0.05, 0.10, -0.02, 0.08, 0.03, 0.07, -0.01, 0.09])
    out = st.summarize(r)
    for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n"):
        assert k in out
    assert out["n"] == 8
    assert 0.0 < out["hit_rate"] < 1.0


def test_summarize_nan_on_single_obs():
    out = st.summarize(pd.Series([0.05]))
    assert np.isnan(out["tstat"])


# ---- the spine: planted premium is recoverable, null is not ------------------
def test_anomaly_beats_null():
    """With premium > 0 the mean spread is materially positive; at null it is near zero."""
    lev0, fwd0, _ = _make(premium=0.0, seed=7)
    lev1, fwd1, _ = _make(premium=0.10, seed=7)

    s0 = st.summarize(st.quintile_spread(lev0, fwd0)["spread"])
    s1 = st.summarize(st.quintile_spread(lev1, fwd1)["spread"])
    assert s1["mean"] > s0["mean"]
    assert s1["mean"] > 0.04   # planted premium is recovered


def _make(premium, seed):
    from leverage_anomaly import data
    return data.synthetic_panel(n_firms=250, n_years=18, premium=premium, seed=seed)
