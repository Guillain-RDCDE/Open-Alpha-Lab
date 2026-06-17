"""Signals, lead-lag regression, directional rule, and the study's spine:
the slope is significant only when a lead-lag is planted; on the null it is zero."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from twitter_mood import data, strategy as st  # noqa: E402

CACHE_PATH = data.INDEX_CACHE


# ---------------------------------------------------------------------------
# Lead-lag regression
# ---------------------------------------------------------------------------
def test_leadlag_keys(null_panel):
    df, _ = null_panel
    r = st.leadlag_regression(df, lag=1)
    assert set(["beta", "tstat", "r2", "n", "lag"]).issubset(r)


def test_leadlag_no_lookahead(null_panel):
    """The response is the FUTURE return; shifting must drop the last lag rows."""
    df, _ = null_panel
    r1 = st.leadlag_regression(df, lag=1)
    r3 = st.leadlag_regression(df, lag=3)
    assert r3["n"] < r1["n"], "Larger lag must use fewer aligned observations"


def test_leadlag_deterministic(live_panel):
    df, _ = live_panel
    a = st.leadlag_regression(df, lag=1)
    b = st.leadlag_regression(df, lag=1)
    assert a == b


def test_sweep_shape(null_panel):
    df, _ = null_panel
    sw = st.leadlag_sweep(df, max_lag=5)
    assert len(sw) == 5
    assert list(sw["lag"]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Directional rule & strategy
# ---------------------------------------------------------------------------
def test_mood_strategy_columns(null_panel):
    df, _ = null_panel
    res = st.mood_strategy(df, lookback=20)
    assert set(["pos", "gross", "net", "mkt"]).issubset(res.columns)
    assert len(res) > 0


def test_mood_strategy_net_le_gross(live_panel):
    """Costs can only subtract: net <= gross every day."""
    df, _ = live_panel
    res = st.mood_strategy(df, lookback=20, allow_short=True, one_way_bps=5.0)
    assert (res["net"] <= res["gross"] + 1e-12).all()


def test_mood_strategy_long_only_positions(null_panel):
    df, _ = null_panel
    res = st.mood_strategy(df, lookback=20, allow_short=False)
    assert set(np.unique(res["pos"].dropna())).issubset({0.0, 1.0})


def test_mood_strategy_short_positions(null_panel):
    df, _ = null_panel
    res = st.mood_strategy(df, lookback=20, allow_short=True)
    assert set(np.unique(res["pos"].dropna())).issubset({-1.0, 1.0})


def test_directional_hitrate_keys(null_panel):
    df, _ = null_panel
    hr = st.directional_hitrate(df, lookback=20)
    assert set(["hit_rate", "uncond_up_rate", "n"]).issubset(hr)
    assert 0.0 <= hr["hit_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(60) * 0.01)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.01, 0.005, 200))
    s = st.summarize(r)
    assert s["tstat"] > 0


def test_annualise_daily():
    s = st.summarize(pd.Series([0.001] * 252))
    out = st.annualise_daily(s, periods=252)
    assert abs(out["mean_ann"] - 252 * 0.001) < 1e-9


# ---------------------------------------------------------------------------
# Shuffle control
# ---------------------------------------------------------------------------
def test_shuffle_null_reproducible(null_panel):
    df, _ = null_panel
    a = st.shuffle_null(df, lag=1, n_shuffles=100, seed=1)
    b = st.shuffle_null(df, lag=1, n_shuffles=100, seed=1)
    assert a == b


def test_shuffle_null_keys(live_panel):
    df, _ = live_panel
    s = st.shuffle_null(df, lag=1, n_shuffles=100, seed=256)
    assert set(["real_beta", "perm_p", "n_shuffles"]).issubset(s)


# ---------------------------------------------------------------------------
# The spine: predictive knob switches the slope on and off
# ---------------------------------------------------------------------------
def test_predictive_knob_significant(live_panel):
    """With a planted lead-lag, the lag-1 slope is positive and significant (t>2)."""
    df, _ = live_panel
    r = st.leadlag_regression(df, lag=1)
    assert r["beta"] > 0.0, f"Planted lead-lag should be positive, got {r['beta']:.5f}"
    assert r["tstat"] > 2.0, f"Planted lead-lag should be significant, got t={r['tstat']:.2f}"


def test_null_knob_insignificant(null_panel):
    """On the null panel the lag-1 slope is statistically zero (|t| < 2)."""
    df, _ = null_panel
    r = st.leadlag_regression(df, lag=1)
    assert abs(r["tstat"]) < 2.0, f"Null slope should be insignificant, got t={r['tstat']:.2f}"


def test_null_shuffle_pvalue_large(null_panel):
    """On the null, the permutation p-value should be far from significance."""
    df, _ = null_panel
    s = st.shuffle_null(df, lag=1, n_shuffles=500, seed=256)
    assert s["perm_p"] > 0.10, f"Null perm_p should be large, got {s['perm_p']:.3f}"


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_daily.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_leadlag_is_float():
    df = data.build_real_panel(fetch=False, cache_path=CACHE_PATH)
    r = st.leadlag_regression(df, lag=1)
    assert isinstance(r["tstat"], float)
    assert np.isfinite(r["tstat"])


@requires_cache
def test_real_strategy_runs():
    df = data.build_real_panel(fetch=False, cache_path=CACHE_PATH)
    res = st.mood_strategy(df, lookback=20, allow_short=True)
    assert len(res) > 50
    assert res["net"].notna().all()
