"""Quintile sort mechanics, HAC inference, and the study's spine:
the ILLIQ sort only beats a random control when a premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amihud_illiquidity import strategy as st  # noqa: E402


# ---- log-ILLIQ transformation ---------------------------------------------
def test_log_illiq_positive_input(null_panel):
    illiq, _, _ = null_panel
    log_i = st.log_illiq(illiq)
    # log of positive number is finite and can be negative
    assert log_i.notna().all().all()
    assert np.isfinite(log_i.values).all()


def test_log_illiq_zero_becomes_nan():
    df = pd.DataFrame({"A": [1.0, 0.0], "B": [2.0, 1.0]},
                      index=pd.Index([2010, 2011], name="year"))
    log_i = st.log_illiq(df)
    assert np.isnan(log_i.loc[2011, "A"])
    assert np.isfinite(log_i.loc[2010, "A"])


# ---- quintile sort --------------------------------------------------------
def test_quintile_sort_shape_and_columns(null_panel):
    illiq, fwd_ret, _ = null_panel
    res = st.quintile_sort(illiq, fwd_ret, n_quintiles=5)
    expected_cols = {"q1", "q2", "q3", "q4", "q5", "hedge", "market", "n", "n_q"}
    assert expected_cols.issubset(set(res.columns))
    assert len(res) >= 10   # at least most of the 20 years pass the filter


def test_quintile_sort_hedge_is_q5_minus_q1(null_panel):
    illiq, fwd_ret, _ = null_panel
    res = st.quintile_sort(illiq, fwd_ret, n_quintiles=5)
    # hedge == q5 - q1 (within floating point)
    diff = (res["q5"] - res["q1"] - res["hedge"]).abs()
    assert (diff < 1e-10).all()


def test_quintile_sort_n_is_total_firms(null_panel):
    illiq, fwd_ret, truth = null_panel
    res = st.quintile_sort(illiq, fwd_ret, n_quintiles=5)
    # n should be close to n_firms (some may drop if no fwd_ret)
    assert (res["n"] > truth["n_firms"] * 0.5).all()


# ---- random control -------------------------------------------------------
def test_random_hedge_returns_shape(null_panel):
    illiq, fwd_ret, _ = null_panel
    rand = st.random_hedge_returns(illiq, fwd_ret, n_draws=20, seed=42)
    assert len(rand) > 0
    assert rand.name == "random_hedge"


def test_random_hedge_reproducible(null_panel):
    illiq, fwd_ret, _ = null_panel
    a = st.random_hedge_returns(illiq, fwd_ret, n_draws=20, seed=1)
    b = st.random_hedge_returns(illiq, fwd_ret, n_draws=20, seed=1)
    assert np.allclose(a.values, b.values)
    c = st.random_hedge_returns(illiq, fwd_ret, n_draws=20, seed=2)
    assert not np.allclose(a.values, c.values)


# ---- summarize ------------------------------------------------------------
def test_summarize_keys():
    s = st.summarize(pd.Series([0.05, -0.03, 0.07, 0.02, -0.01, 0.04]))
    for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n"):
        assert k in s


def test_summarize_tstat_finite_for_enough_obs():
    r = pd.Series(np.random.default_rng(0).normal(0.02, 0.15, 20))
    s = st.summarize(r)
    assert np.isfinite(s["tstat"])


# ---- the spine: premium planted -> sort finds it; null -> random does as well ----
def test_sort_beats_random_only_with_premium(null_panel, premium_panel):
    """The quintile sort finds an edge only when the premium is planted."""
    def mean_hedge(illiq, fwd_ret):
        res = st.quintile_sort(illiq, fwd_ret)
        return float(res["hedge"].mean())

    def mean_random(illiq, fwd_ret):
        rand = st.random_hedge_returns(illiq, fwd_ret, n_draws=200, seed=42)
        return float(rand.mean())

    illiq_p, fwd_p, _ = premium_panel
    illiq_n, fwd_n, _ = null_panel

    # With premium: ILLIQ hedge >> random control
    assert mean_hedge(illiq_p, fwd_p) > mean_random(illiq_p, fwd_p) + 0.01

    # Without premium: ILLIQ hedge is close to random (within 4ppt)
    hedge_null = mean_hedge(illiq_n, fwd_n)
    rand_null = mean_random(illiq_n, fwd_n)
    assert abs(hedge_null - rand_null) < 0.04
