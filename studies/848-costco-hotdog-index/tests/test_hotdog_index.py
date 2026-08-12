"""Fully offline, deterministic tests for Study 848 ("Costco Hot-Dog Index").

No network: every test runs on the embedded real CPI values (BLS `CUSR0000SA0`, the
offline fallback for the cached BLS pull) or the fixed-seed synthetic world. We assert
(1) the CPI series has the documented shape (continuous monthly grid, the 2021-23 surge),
(2) the real-price identity is arithmetic and shows a
large erosion, (3) the HAC regression matches numpy's closed-form OLS point estimate,
(4) the synthetic positive control recovers a planted inflation beta while the null world
stays flat, (5) the beta-gap / placebo / era-cut / timer plumbing is internally
consistent, and (6) the inference primitives behave. A single real-cache test is guarded
so the suite passes with NO cache present.

    pytest -q studies/848-costco-hotdog-index/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from hotdog_index import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The CPI series (real BLS values)
# --------------------------------------------------------------------------- #
def test_cpi_series_shape():
    cpi = data.load_cpi()
    months = pd.PeriodIndex(cpi.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    assert len(months) == len(expected)
    assert not expected.difference(months).size      # continuous monthly grid, no holes
    assert cpi.is_monotonic_increasing or (cpi.diff().dropna() > -5).all()  # broadly rising
    # the 2021-23 surge: YoY inflation peaks near 9% in mid-2022
    yoy = cpi / cpi.shift(12) - 1.0
    assert 0.07 < yoy.max() < 0.11
    # the level roughly doubles from 2000 to 2026
    assert 1.8 < cpi.iloc[-1] / cpi.iloc[0] < 2.1


# --------------------------------------------------------------------------- #
# The real-price identity
# --------------------------------------------------------------------------- #
def test_real_price_is_mechanical_identity():
    cpi = data.load_cpi()
    rp = st.real_price(cpi, nominal=1.50, base_cpi=data.CPI_1985)
    # identity: real_price * cpi / base == nominal, exactly, every month
    assert np.allclose((rp * cpi / data.CPI_1985).to_numpy(), 1.50, atol=1e-9)
    er = st.erosion_stats(cpi, nominal=1.50, base_cpi=data.CPI_1985)
    assert 50.0 < er["eroded_pct"] < 80.0            # a large (but bounded) erosion
    assert er["real_now"] < 1.50                     # real value below the frozen nominal
    assert er["inflation_adjusted"] > 3.0            # would cost >$3 to hold 1985 value
    assert er["giveaway"] == pytest.approx(er["inflation_adjusted"] - 1.50, abs=1e-9)


# --------------------------------------------------------------------------- #
# HAC regression
# --------------------------------------------------------------------------- #
def test_hac_regression_matches_numpy_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = 1.5 + 0.7 * x + rng.standard_normal(300)
    r = st.hac_regression(y, x, lags=6)
    X = np.column_stack([np.ones(300), x])
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose([r["alpha"], r["slope"]], b_ols, atol=1e-10)
    assert abs(r["slope"] - 0.7) < 0.15
    assert r["t_nw"] > 8                              # a strong true slope is significant


def test_hac_regression_null_slope_insignificant():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400)
    y = rng.standard_normal(400)                     # y independent of x
    r = st.hac_regression(y, x, lags=6)
    assert abs(r["t_nw"]) < 2.5


# --------------------------------------------------------------------------- #
# Synthetic control — recovers a planted inflation beta; null stays flat
# --------------------------------------------------------------------------- #
def test_synthetic_world_deterministic(planted_panel):
    p2 = data.synthetic_world(edge=8.0, seed=848)
    assert np.allclose(planted_panel.to_numpy(dtype=float), p2.to_numpy(dtype=float),
                       equal_nan=True)
    # kept a PeriodIndex (no ns-Timestamp overflow)
    assert isinstance(planted_panel.index, pd.PeriodIndex)


def test_planted_inflation_beta_recovered(planted_panel):
    cb = st.inflation_beta(planted_panel, "cost", lag=0)
    assert cb["slope"] > 0                            # positive inflation beta
    assert cb["t_nw"] > 3.0                           # the regression lights up
    gap = st.beta_gap(planted_panel, lag=0)
    assert gap["cost_minus_xlp"]["t_nw"] > 2.5        # distinctively above staples


def test_null_world_no_signal(null_panel):
    cb = st.inflation_beta(null_panel, "cost", lag=0)
    assert abs(cb["t_nw"]) < 2.5                      # detector stays silent on the null


def test_null_placebo_centered(null_panel):
    pl = st.placebo_pvalue(null_panel, "cost", n_perm=600)
    assert 0.05 < pl["p_value"] < 0.95               # observed sits inside the null cloud
    assert abs(pl["placebo_mean"]) < abs(pl["placebo_sd"])


def test_seed_robust_control():
    null = st.synthetic_mean_t(data, edge=0.0, n_seeds=8)
    planted = st.synthetic_mean_t(data, edge=8.0, n_seeds=8)
    assert abs(null["mean_t"]) < 1.5                  # no false positive on average
    assert planted["mean_t"] > 3.0                    # planted edge recovered across seeds
    assert planted["mean_slope"] > null["mean_slope"]


# --------------------------------------------------------------------------- #
# Total-return race + timer plumbing
# --------------------------------------------------------------------------- #
def test_total_return_race_consistent(planted_panel):
    race = st.total_return_race(planted_panel)
    for k in ("cost", "spy", "cpi"):
        assert race[k]["mult"] > 0
        # real multiple == nominal multiple / CPI multiple
        assert race[k]["real_mult"] == pytest.approx(race[k]["mult"] / race["cpi"]["mult"],
                                                     rel=1e-9)
    assert race["cpi"]["real_mult"] == pytest.approx(1.0)


def test_timer_runs_and_costs_hurt(planted_panel):
    cheap = st.timer_stats(planted_panel, cost_bps=0.0)
    dear = st.timer_stats(planted_panel, cost_bps=100.0)
    assert dear["timer_sharpe"] <= cheap["timer_sharpe"] + 1e-9   # more cost -> weakly worse
    assert 0.0 <= dear["invested_frac"] <= 1.0


def test_era_cut_shape(planted_panel):
    tab = st.era_cut(planted_panel, split="2013-01-01")
    assert list(tab.index) == ["early", "late"]
    assert tab["n"].min() > 20


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_inference_primitives():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert st.welch_t(a, b) > 5
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_inflation_beta_is_deterministic(planted_panel):
    a = st.inflation_beta(planted_panel, "cost", lag=0)["t_nw"]
    b = st.inflation_beta(planted_panel, "cost", lag=0)["t_nw"]
    assert a == b                                     # pure function, no hidden randomness


# --------------------------------------------------------------------------- #
# Real cache — only if present (guarded; offline CI skips it)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH),
                    reason="real cache absent offline CI")
def test_real_cache_loads_and_builds():
    panel = data.build_panel()
    assert len(panel) > 200
    assert {"cost", "spy", "xlp", "cpi", "surprise"} <= set(panel.columns)
    er = st.erosion_stats(data.load_cpi())
    assert er["eroded_pct"] > 50.0
    cb = st.inflation_beta(panel, "cost", lag=0)
    assert np.isfinite(cb["t_nw"])
