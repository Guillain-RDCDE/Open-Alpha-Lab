"""Strategy tests for Study 973 — a known correlation, a planted delay, and the fixes."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from calendar_gap import data, strategy as st  # noqa: E402


def _pair(rho=0.7, delay=0.0, n=6000, seed=973, vol=0.01):
    """Two assets on one common factor; ``delay`` is the share of the factor that the second
    asset only sees the following day (stale pricing)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, vol, n)
    e1 = rng.normal(0, vol * np.sqrt(1 / rho ** 2 - 1), n)
    e2 = rng.normal(0, vol * np.sqrt(1 / rho ** 2 - 1), n)
    a = f + e1
    f_lag = np.concatenate([[0.0], f[:-1]])
    b = (1 - delay) * f + delay * f_lag + e2
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"A": 100 * np.cumprod(1 + a), "B": 100 * np.cumprod(1 + b)},
                        index=idx)


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #
def test_synchronous_pair_has_no_bias():
    px = _pair(delay=0.0)
    agg = st.aggregated_correlation(px, "A", "B")
    assert agg.loc["daily", "correlation"] == pytest.approx(
        agg.loc["monthly", "correlation"], abs=0.10)


def test_a_planted_delay_depresses_the_daily_correlation():
    sync = _pair(delay=0.0)
    stale = _pair(delay=0.5)
    c_sync = st.to_returns(sync).corr().loc["A", "B"]
    c_stale = st.to_returns(stale).corr().loc["A", "B"]
    assert c_stale < c_sync - 0.15


def test_lower_frequency_recovers_the_correlation():
    px = _pair(delay=0.5)
    agg = st.aggregated_correlation(px, "A", "B")
    assert agg.loc["monthly", "correlation"] > agg.loc["daily", "correlation"] + 0.15
    assert agg["correlation"].is_monotonic_increasing


def test_lead_lag_profile_points_at_the_delay():
    px = _pair(delay=0.6)
    prof = st.lead_lag_profile(st.to_returns(px), "A", "B", max_lag=2)
    assert prof.idxmax() in (0, 1)
    assert prof[1] > prof[-1]      # B follows A, not the other way round


def test_lead_lag_profile_is_symmetric_when_synchronous():
    prof = st.lead_lag_profile(st.to_returns(_pair(delay=0.0)), "A", "B", max_lag=2)
    assert prof.idxmax() == 0
    assert abs(prof[1] - prof[-1]) < 0.06


# --------------------------------------------------------------------------- #
# The corrections
# --------------------------------------------------------------------------- #
def test_dimson_recovers_the_beta_a_delay_destroys():
    px = _pair(delay=0.5)
    r = st.to_returns(px)
    naive = st.ols_beta(r["B"], r["A"])
    dim = st.dimson_beta(r["B"], r["A"], n_lags=1)["beta"]
    assert dim > naive + 0.15


def test_dimson_with_zero_lags_is_the_ordinary_beta():
    r = st.to_returns(_pair(delay=0.3))
    assert st.dimson_beta(r["B"], r["A"], n_lags=0)["beta"] == pytest.approx(
        st.ols_beta(r["B"], r["A"]), rel=1e-6)


def test_dimson_leaves_a_synchronous_beta_alone():
    r = st.to_returns(_pair(delay=0.0))
    naive = st.ols_beta(r["B"], r["A"])
    dim = st.dimson_beta(r["B"], r["A"], n_lags=1)["beta"]
    assert dim == pytest.approx(naive, abs=0.08)


def test_scholes_williams_agrees_with_dimson_in_direction():
    r = st.to_returns(_pair(delay=0.5))
    naive = st.ols_beta(r["B"], r["A"])
    sw = st.scholes_williams_beta(r["B"], r["A"])["beta"]
    dim = st.dimson_beta(r["B"], r["A"], n_lags=1)["beta"]
    assert sw > naive
    assert abs(sw - dim) < 0.30


def test_the_lagged_coefficient_is_the_smoking_gun():
    r = st.to_returns(_pair(delay=0.5))
    d = st.dimson_beta(r["B"], r["A"], n_lags=1)
    assert d["lag1"] > 0.15        # yesterday's A still explains today's B
    assert abs(d["lead1"]) < 0.15  # tomorrow's A does not


# --------------------------------------------------------------------------- #
# Portfolio consequences
# --------------------------------------------------------------------------- #
def test_min_variance_weights_sum_to_one():
    cov = np.array([[4e-4, 1e-4], [1e-4, 9e-4]])
    w = st.min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]          # the quieter asset gets more


def test_biased_matrix_understates_portfolio_volatility():
    px = _pair(delay=0.6, n=8000)
    imp = st.portfolio_impact(px, ["A", "B"], step_estimate=1, step_truth=21)
    assert imp["delivered_vol"] > imp["promised_vol"]
    assert imp["understatement"] > 0.02


def test_no_understatement_when_the_pair_is_synchronous():
    px = _pair(delay=0.0, n=8000)
    imp = st.portfolio_impact(px, ["A", "B"], step_estimate=1, step_truth=21)
    assert abs(imp["understatement"]) < 0.10


def test_to_returns_step_gives_non_overlapping_observations():
    px = _pair(n=1000)
    r5 = st.to_returns(px, 5)
    assert len(r5) == pytest.approx(len(px) / 5, rel=0.05)


def test_bias_table_has_every_column(planted):
    prices, _, _ = planted
    cols = list(prices.columns)
    t = st.bias_table(prices, cols[0], cols[1:3])
    assert {"corr_daily", "beta_naive", "beta_dimson", "beta_sw", "corr_lift"} <= set(t.columns)
    assert len(t) == 2


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_big_lifts": 3, "n_foreign": 4, "control_lift": 0.01, "control_asset": "IWM",
         "worst_lift_asset": "EWJ", "worst_daily_corr": 0.55, "worst_monthly_corr": 0.78,
         "worst_lift": 0.23, "worst_beta_naive": 0.6, "worst_beta_dimson": 0.85,
         "worst_lag_coef": 0.22, "promised_vol": 0.11, "delivered_vol": 0.125,
         "understatement": 0.14, "max_weight_gap": 0.22}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(control_lift=0.20))["signal"] == "Weak"
    assert st.verdict(_headline(n_big_lifts=0))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(understatement=0.02))["trad"] == "Fragile"
    assert st.verdict(_headline(understatement=0.001))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(worst_lift=0.31))
    assert "+0.31" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
