"""Signals, regression, tercile sort, timing overlay, and the study's spine:
the predictive regression fires only when a contrarian effect is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insider_buying import data, strategy as st  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(60) * 0.04)
    s = st.summarize(r)
    for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n", "mean_ann"):
        assert k in s


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.05, 0.03, 120))
    assert st.summarize(r)["tstat"] > 0


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_regression_deterministic(null_panel):
    df, _ = null_panel
    a = st.predictive_regression(df)
    b = st.predictive_regression(df)
    assert a["tstat"] == b["tstat"]
    assert a["beta"] == b["beta"]


def test_regression_keys(null_panel):
    df, _ = null_panel
    reg = st.predictive_regression(df)
    for k in ("beta", "alpha", "tstat", "r2", "n"):
        assert k in reg


def test_regression_null_insignificant(null_panel):
    """On the null panel the slope t-stat must be small (no planted effect)."""
    df, _ = null_panel
    reg = st.predictive_regression(df)
    assert np.isfinite(reg["tstat"])
    assert abs(reg["tstat"]) < 2.0, f"Null t too large: {reg['tstat']:.2f}"


# ---------------------------------------------------------------------------
# The spine: planted beta is detected
# ---------------------------------------------------------------------------
def test_planted_beta_detected(live_panel):
    """With a planted contrarian beta, the regression slope is positive and t > 2."""
    df, _ = live_panel
    reg = st.predictive_regression(df)
    assert reg["beta"] > 0.0, f"Expected positive slope, got {reg['beta']:.4f}"
    assert reg["tstat"] > 2.0, f"Expected t>2 on planted effect, got {reg['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Tercile sort
# ---------------------------------------------------------------------------
def test_tercile_structure(null_panel):
    df, _ = null_panel
    terc = st.tercile_returns(df)
    assert list(terc.index) == ["low", "mid", "high", "high_minus_low"]
    assert set(["mean_ann", "tstat", "n"]).issubset(terc.columns)


def test_tercile_high_minus_low_sign(live_panel):
    """With a planted contrarian beta, high-ratio months should be followed by
    HIGHER returns -> high_minus_low > 0."""
    df, _ = live_panel
    terc = st.tercile_returns(df)
    assert terc.loc["high_minus_low", "mean_ann"] > 0.0


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def test_overlay_columns(null_panel):
    df, _ = null_panel
    ov = st.timing_overlay(df, lookback=24, one_way_bps=5.0)
    assert set(["pos", "gross", "cost", "net", "bah"]).issubset(ov.columns)


def test_overlay_position_binary(null_panel):
    df, _ = null_panel
    ov = st.timing_overlay(df, lookback=24)
    assert set(np.unique(ov["pos"].to_numpy())).issubset({0.0, 1.0})


def test_overlay_costs_nonnegative(null_panel):
    df, _ = null_panel
    ov = st.timing_overlay(df, lookback=24, one_way_bps=10.0)
    assert (ov["cost"] >= 0).all()
    # net <= gross always (costs only subtract)
    assert (ov["net"] <= ov["gross"] + 1e-12).all()


def test_overlay_no_lookahead_threshold(null_panel):
    """The trailing-median threshold must use only past ratios: the first row's
    position defaults to long (threshold undefined) and never peeks ahead."""
    df, _ = null_panel
    ov = st.timing_overlay(df, lookback=24)
    assert ov["pos"].iloc[0] == 1.0


def test_overlay_turnover_range(null_panel):
    df, _ = null_panel
    ov = st.timing_overlay(df, lookback=24)
    t = st.overlay_turnover(ov)
    assert 0.0 <= t <= 2.0


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regression_is_float():
    g = data.fetch_sp500(fetch=False, cache_path=CACHE_PATH)
    df = data.build_real_panel(g)
    reg = st.predictive_regression(df)
    assert isinstance(reg["tstat"], float) and np.isfinite(reg["tstat"])


@requires_cache
def test_real_overlay_underperforms_bah():
    """On the real tape, the long/flat overlay should not beat buy-and-hold."""
    g = data.fetch_sp500(fetch=False, cache_path=CACHE_PATH)
    df = data.build_real_panel(g)
    ov = st.timing_overlay(df, lookback=24, one_way_bps=5.0)
    net = st.summarize(ov["net"])["mean_ann"]
    bah = st.summarize(ov["bah"])["mean_ann"]
    assert net < bah, "Overlay should underperform buy-and-hold on the real tape"
