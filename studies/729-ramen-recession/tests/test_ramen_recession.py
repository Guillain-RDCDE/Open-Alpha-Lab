"""Fully offline, deterministic tests for Study 729 (the ramen index).

No network, no cache dependency: every test runs either on a fixed-seed synthetic control or
on tiny hand-built series with a *known* answer. The engine must (1) recover a planted lead
with the lead-lag cross-correlation, (2) find no lead when there is none, (3) recover a planted
asymmetric (defensive) beta, (4) compute the CAPM/beta/return primitives exactly, and (5)
detect a planted recession-window edge — and none when there isn't one. The hardcoded WINA
series and NBER windows are checked for shape only (no network).
"""

import numpy as np
import pandas as pd

from ramen_recession import data, strategy as st


# --------------------------------------------------------------------------- #
# THE HEADLINE engine — lead-lag cross-correlation
# --------------------------------------------------------------------------- #
def test_lead_lag_recovers_a_planted_lead():
    ig, mk = data.synthetic_leading_index(lead=1, seed=729)
    cl = st.control_recovers_lead(ig, mk, planted_lead=1)
    assert cl["best_lead"] == 1                 # the planted lead is recovered
    assert cl["best_r"] < 0                      # a real "tell" is a negative correlation
    assert cl["best_t"] < -2.0                   # and significant
    assert cl["recovered_lead_ok"] == 1


def test_lead_lag_finds_no_lead_in_pure_noise():
    rng = np.random.default_rng(11)
    yrs = pd.date_range("1990-12-31", periods=25, freq="YE")
    a = pd.Series(rng.normal(0, 1, 25), index=yrs)
    b = pd.Series(rng.normal(0, 1, 25), index=yrs)   # independent of a
    ll = st.lead_lag_corr(a, b, leads=range(-2, 3))
    # no lead should clear |t| >= 2 on independent noise
    assert all(abs(v["t"]) < 2.0 for v in ll["per_lead"].values() if np.isfinite(v["t"]))


# --------------------------------------------------------------------------- #
# Defensive engine — bull/bear beta
# --------------------------------------------------------------------------- #
def test_synthetic_defensive_recovers_asymmetry():
    mkt, stock = data.synthetic_defensive(beta_down=0.5, beta_up=1.0, seed=7290)
    bb = st.bull_bear_beta(stock, mkt, split=0.0)
    assert bb["down_beta"] < bb["up_beta"]       # defensive shape recovered
    assert bb["down_beta"] < 1.0
    assert bb["defensive"] == 1


def test_symmetric_beta_is_not_flagged_defensive():
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.0, 0.045, 800))
    stock = 0.9 * mkt + pd.Series(rng.normal(0, 0.03, 800))
    bb = st.bull_bear_beta(stock, mkt, split=0.0)
    assert abs(bb["asymmetry"]) < 0.15           # near-symmetric -> not defensive by design


# --------------------------------------------------------------------------- #
# Return / risk primitives — exact on a controlled input
# --------------------------------------------------------------------------- #
def test_cagr_exact_on_known_doubling():
    idx = pd.to_datetime(["2000-01-01", "2002-01-01", "2004-01-01"])
    lvl = pd.Series([100.0, 200.0, 400.0], index=idx)
    assert abs(st.cagr(lvl) - (2 ** 0.5 - 1)) < 0.02


def test_max_drawdown_sign_and_value():
    lvl = pd.Series([100.0, 120.0, 60.0, 90.0])  # peak 120 -> trough 60 = -50%
    assert abs(st.max_drawdown(lvl) - (-0.5)) < 1e-9


def test_beta_recovered_on_pure_linear_relation():
    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0, 0.04, 300))
    stock = 1.3 * mkt                            # exact beta 1.3, no noise
    nw = st.newey_west_alpha_t(stock, mkt, lags=6)
    assert abs(nw["beta"] - 1.3) < 1e-6
    assert abs(nw["alpha_m"]) < 1e-6


# --------------------------------------------------------------------------- #
# Recession-window test — detects a planted edge, and a null when there is none
# --------------------------------------------------------------------------- #
def _mask_first_half(index):
    n = len(index)
    return np.array([i < n // 2 for i in range(n)])


def test_recession_excess_detects_planted_edge():
    n = 120
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(2)
    spy = pd.Series(rng.normal(0.0, 0.03, n), index=idx)
    stock = spy.copy()
    stock.iloc[: n // 2] += 0.05                 # +5%/mo edge in the "recession" half
    r = st.recession_excess_t(stock, spy, _mask_first_half)
    assert r["mean_excess"] > 0 and r["t"] > 2.0


def test_recession_excess_null_when_no_edge():
    n = 120
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(0.0, 0.03, n), index=idx)
    stock = spy + pd.Series(rng.normal(0.0, 0.03, n), index=idx)
    r = st.recession_excess_t(stock, spy, _mask_first_half)
    assert abs(r["t"]) < 2.0


# --------------------------------------------------------------------------- #
# Hardcoded series plumbing — pure, no network
# --------------------------------------------------------------------------- #
def test_ramen_index_is_monotone_dates_and_grows_secularly():
    ri = data.load_ramen_index()
    assert ri.index.is_monotonic_increasing
    assert ri.iloc[-1] > ri.iloc[0]              # secular growth over 2005->2024
    g = data.ramen_growth()
    # the honest counter-exhibit: demand FELL in the 2008 GFC year
    assert g[g.index.year == 2008].iloc[0] < 0


def test_recession_months_cover_the_three_recessions():
    months = data.recession_months()
    assert len(months) == 31                     # 2001 (9) + GFC (19) + COVID (3)
    assert pd.Timestamp("2008-10-01") in months
    assert pd.Timestamp("2005-01-01") not in months


def test_recession_years_set():
    ys = data.recession_years()
    assert {2001, 2008, 2020}.issubset(ys)
    assert 2005 not in ys
