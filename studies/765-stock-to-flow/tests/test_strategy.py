"""Offline tests for the S2F strategy + inference: the fit, the residual, the HAC regression,
the timer's honesty rails, and the synthetic control's null/planted behaviour."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_to_flow import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# The fit + residual
# ---------------------------------------------------------------------------
def test_fit_recovers_exact_power_law():
    """On a clean world where ln(price) = 0.5 + 3*ln(SF) exactly, fit_s2f recovers a=0.5, b=3, R2=1."""
    idx = pd.date_range("2015-01-01", periods=400, freq="D")
    sf = pd.Series(np.linspace(20.0, 120.0, 400), index=idx)
    price = np.exp(0.5 + 3.0 * np.log(sf))
    df = pd.DataFrame({"price": price, "supply": 19e6, "flow": 3.3e5, "sf": sf})
    fit = st.fit_s2f(df)
    assert abs(fit["a"] - 0.5) < 1e-6
    assert abs(fit["b"] - 3.0) < 1e-6
    assert fit["r2"] > 0.999999


def test_valuation_residual_sign(planted_world):
    """Residual > 0 exactly where price is above the fitted model line, < 0 below."""
    fit = st.fit_s2f(planted_world)
    a, b = fit["a"], fit["b"]
    resid = st.valuation_residual(planted_world, a, b)
    model = st.model_price(planted_world, a, b)
    above = planted_world["price"] > model
    assert (resid[above] > 0).all()
    assert (resid[~above] < 0).all()


def test_model_price_positive(planted_world):
    fit = st.fit_s2f(planted_world)
    mp = st.model_price(planted_world, fit["a"], fit["b"])
    assert (mp > 0).all()


def test_oos_fit_stats_keys(planted_world):
    o = st.oos_fit_stats(planted_world, train_end="2018-01-01")
    for k in ["a", "b", "r2_in", "r2_oos", "rmse_in", "rmse_oos", "n_train", "n_oos"]:
        assert k in o
    assert o["n_train"] > 0 and o["n_oos"] > 0
    assert np.isfinite(o["r2_in"]) and np.isfinite(o["r2_oos"])


def test_spurious_race_keys(planted_world):
    race = st.spurious_trend_race(planted_world)
    assert set(["r2_sf", "r2_time", "corr_sf_time"]).issubset(race)
    assert -1.0 <= race["corr_sf_time"] <= 1.0


# ---------------------------------------------------------------------------
# HAC inference
# ---------------------------------------------------------------------------
def test_newey_west_recovers_known_slope():
    """A clean linear y = 2x + noise: NW slope ~ 2 with a large |t|."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 2.0 * x + 0.1 * rng.standard_normal(500)
    nw = st.newey_west_t(x, y, lag=5)
    assert abs(nw["slope"] - 2.0) < 0.1
    assert abs(nw["t"]) > 5


def test_newey_west_null_small_t():
    """Independent x, y: slope ~ 0, |t| small."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(600)
    y = rng.standard_normal(600)
    nw = st.newey_west_t(x, y, lag=5)
    assert abs(nw["t"]) < 3.0


def test_predictive_regression_columns(planted_world):
    fit = st.fit_s2f(planted_world)
    resid = st.valuation_residual(planted_world, fit["a"], fit["b"])
    pr = st.predictive_regression(planted_world, resid, horizons=(30, 90))
    assert set(["slope", "hac_t", "n"]).issubset(pr.columns)
    assert list(pr.index) == [30, 90]


# ---------------------------------------------------------------------------
# Timer honesty rails
# ---------------------------------------------------------------------------
def test_timer_zero_cost_gross_equals_net(planted_world):
    fit = st.fit_s2f(planted_world)
    resid = st.valuation_residual(planted_world, fit["a"], fit["b"])
    tb = st.timer_backtest(planted_world, resid, cost_bps=0.0)
    assert abs(tb["gross_total_pct"] - tb["net_total_pct"]) < 1e-6


def test_timer_costs_reduce_return(planted_world):
    fit = st.fit_s2f(planted_world)
    resid = st.valuation_residual(planted_world, fit["a"], fit["b"])
    tb = st.timer_backtest(planted_world, resid, cost_bps=25.0)
    assert tb["net_total_pct"] <= tb["gross_total_pct"] + 1e-9


def test_timer_exposure_bounds(planted_world):
    fit = st.fit_s2f(planted_world)
    resid = st.valuation_residual(planted_world, fit["a"], fit["b"])
    tb = st.timer_backtest(planted_world, resid)
    assert 0.0 <= tb["exposure_pct"] <= 100.0


def test_timer_execution_lag_no_lookahead():
    """The position must act on YESTERDAY's residual: a residual that turns cheap on day t can
    only earn from day t+1, never the day-t return itself."""
    idx = pd.date_range("2015-01-01", periods=6, freq="D")
    price = pd.Series([100, 100, 100, 200, 200, 200], index=idx, dtype=float)
    sf = pd.Series([50.0] * 6, index=idx)
    df = pd.DataFrame({"price": price, "supply": 19e6, "flow": price * 0 + 3.3e5, "sf": sf})
    # residual: rich (>0) until day index 2, then cheap (<0) from day 3 onward
    resid = pd.Series([1, 1, 1, -1, -1, -1], index=idx, dtype=float)
    tb = st.timer_backtest(df, resid, threshold=0.0, cost_bps=0.0)
    # The 100->200 jump happens on day index 3. The residual only turns cheap ON day 3, so with a
    # one-day lag the position is still 0 that day and MISSES the jump -> gross ~ 0, not +100%.
    assert tb["gross_total_pct"] < 1.0


# ---------------------------------------------------------------------------
# Synthetic control — null vs planted
# ---------------------------------------------------------------------------
def test_synthetic_null_unbiased():
    """Across 20 seeds the null detector t is centred near zero (no systematic bias)."""
    ts = np.array([st.synthetic_detect(data.synthetic_world(beta=0.0, seed=765 + s))
                   for s in range(20)])
    assert abs(ts.mean()) < 1.0


def test_synthetic_planted_negative():
    """A planted mean-reversion effect => significant NEGATIVE slope (cheap -> higher returns)."""
    t = st.synthetic_detect(data.synthetic_world(beta=0.03, seed=765))
    assert t < -2.0, f"planted effect should give t < -2, got {t:.2f}"
