"""Tests for the strategy layer of Study 267 (M2-Growth).

All tests are offline and deterministic — synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2_growth import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: a merged frame that looks like build_real_panel() output
# ---------------------------------------------------------------------------
def _make_panel(premium: float = 0.0, seed: int = 267, n_months: int = 600) -> pd.DataFrame:
    df, _ = data.synthetic_panel(n_months=n_months, premium=premium, seed=seed)
    df["sp_fwd1"] = df["sp_ret"].shift(-1)
    fwd12 = (1.0 + df["sp_ret"]).shift(-1)
    for k in range(2, 13):
        fwd12 = fwd12 * (1.0 + df["sp_ret"].shift(-k))
    df["sp_fwd12"] = fwd12 - 1.0
    return df


# ---------------------------------------------------------------------------
# Newey-West regression
# ---------------------------------------------------------------------------
def test_nw_regression_keys():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200)
    r = st.nw_regression(y, x, lags=4)
    assert set(r.keys()) == {"beta", "alpha", "t_hac", "t_ols", "r2", "n"}
    assert r["n"] == 200


def test_nw_regression_recovers_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = 2.0 * x + rng.normal(scale=0.5, size=2000)
    r = st.nw_regression(y, x, lags=0)
    assert abs(r["beta"] - 2.0) < 0.1
    assert r["t_hac"] > 5


def test_nw_regression_zero_slope_under_null():
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    r = st.nw_regression(y, x, lags=3)
    assert abs(r["t_hac"]) < 3.0


def test_nw_handles_nans():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    y = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    r = st.nw_regression(y, x, lags=1)
    assert r["n"] == 5


# ---------------------------------------------------------------------------
# Predictive regression: null vs planted
# ---------------------------------------------------------------------------
def test_predictive_null_is_insignificant():
    df = _make_panel(premium=0.0, seed=267, n_months=600)
    r = st.predictive_regression(df, target_col="sp_fwd1", lags=1)
    assert abs(r["t_hac"]) < 3.0


def test_predictive_planted_is_detectable():
    df = _make_panel(premium=2.0, seed=267, n_months=600)
    r = st.predictive_regression(df, target_col="sp_fwd1", lags=1)
    assert r["beta"] > 0
    assert r["t_hac"] > 2.0


def test_hac_tames_overlap_inflation():
    """On overlapping forward windows, |t_OLS| should exceed |t_HAC| (HAC widens SE)."""
    df = _make_panel(premium=1.0, seed=267, n_months=600)
    r = st.predictive_regression(df, target_col="sp_fwd12", lags=18)
    assert abs(r["t_ols"]) >= abs(r["t_hac"])


# ---------------------------------------------------------------------------
# Quantile sort
# ---------------------------------------------------------------------------
def test_quantile_sort_shape():
    df = _make_panel(premium=0.0)
    q = st.quantile_sort(df, n_buckets=5)
    assert len(q) == 5
    assert set(q.columns) == {"mean_fwd_ret", "n", "mean_m2"}


def test_quantile_buckets_monotone_in_m2():
    df = _make_panel(premium=0.0)
    q = st.quantile_sort(df, n_buckets=5)
    assert q["mean_m2"].is_monotonic_increasing


def test_high_minus_low_keys():
    df = _make_panel(premium=0.0)
    hml = st.high_minus_low(df, n_buckets=5)
    assert set(hml.keys()) == {"hi_mean", "lo_mean", "spread", "t_stat", "n_hi", "n_lo"}


def test_high_minus_low_positive_when_planted():
    df = _make_panel(premium=2.0, n_months=800)
    hml = st.high_minus_low(df, n_buckets=5)
    assert hml["spread"] > 0


# ---------------------------------------------------------------------------
# Backtest rule
# ---------------------------------------------------------------------------
def test_backtest_keys():
    df = _make_panel(premium=0.0)
    bt = st.backtest_rule(df)
    required = {"ann_gross", "ann_net", "ann_bah", "sharpe_net", "sharpe_bah",
                "time_in_market", "n_flips", "turnover_per_yr", "n_months"}
    assert required.issubset(set(bt.keys()))


def test_backtest_net_below_gross():
    """Costs make net <= gross."""
    df = _make_panel(premium=0.0)
    bt = st.backtest_rule(df, cost_bps=20.0)
    assert bt["ann_net"] <= bt["ann_gross"] + 1e-9


def test_backtest_time_in_market_is_fraction():
    df = _make_panel(premium=0.0)
    bt = st.backtest_rule(df)
    assert 0.0 <= bt["time_in_market"] <= 1.0


def test_backtest_deterministic():
    df = _make_panel(premium=0.0)
    a = st.backtest_rule(df)
    b = st.backtest_rule(df)
    assert a["ann_net"] == b["ann_net"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_panel(premium=0.0)
    s = st.summarize(df)
    required = {
        "n_months", "beta_fwd1", "t_hac_fwd1", "t_ols_fwd1",
        "beta_fwd12", "t_hac_fwd12", "spread", "spread_t",
        "ann_net", "ann_bah", "sharpe_net", "sharpe_bah", "turnover_per_yr",
    }
    assert required.issubset(set(s.keys()))
