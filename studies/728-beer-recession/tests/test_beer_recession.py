"""Fully offline, deterministic tests for Study 728 (beer defensiveness).

No network, no cache dependency: every test either runs on the fixed-seed synthetic
control or on tiny hand-built series with a *known* answer. The engine must (1) recover a
planted asymmetric (defensive) beta, (2) show near-symmetry on a symmetric-beta null,
(3) compute the CAPM/beta/return primitives exactly on a controlled input, and (4) detect a
planted recession-window edge — and find none when there isn't one.
"""

import numpy as np
import pandas as pd

from beer_recession import data, strategy as st


# --------------------------------------------------------------------------- #
# Synthetic positive control — a genuinely defensive stock
# --------------------------------------------------------------------------- #
def test_synthetic_defensive_recovers_asymmetry():
    mkt, stock = data.synthetic_defensive(beta_down=0.5, beta_up=1.0, seed=728)
    bb = st.bull_bear_beta(stock, mkt, split=0.0)
    assert bb["down_beta"] < bb["up_beta"]          # defensive shape recovered
    assert bb["down_beta"] < 1.0
    assert bb["defensive"] == 1
    # recovered betas land near the planted values (fixed seed)
    assert abs(bb["down_beta"] - 0.5) < 0.15
    assert abs(bb["up_beta"] - 1.0) < 0.35


def test_control_recovers_helper():
    mkt, stock = data.synthetic_defensive(beta_down=0.5, beta_up=1.0, seed=728)
    cr = st.control_recovers(stock, mkt, planted_down=0.5, planted_up=1.0)
    assert cr["recovered_defensive"] == 1


def test_symmetric_beta_is_not_flagged_defensive():
    # A stock with the SAME beta up and down is NOT defensive (a true null).
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.0, 0.045, 800))    # zero-mean -> balanced sign-split
    stock = 0.9 * mkt + pd.Series(rng.normal(0, 0.03, 800))
    bb = st.bull_bear_beta(stock, mkt, split=0.0)
    assert abs(bb["asymmetry"]) < 0.15              # near-symmetric
    # down and up betas both near the single planted beta
    assert abs(bb["down_beta"] - 0.9) < 0.2
    assert abs(bb["up_beta"] - 0.9) < 0.2


# --------------------------------------------------------------------------- #
# Return / risk primitives — exact on a controlled input
# --------------------------------------------------------------------------- #
def test_cagr_exact_on_known_doubling():
    # doubles every ~2 years over exactly 4 years -> CAGR = 2^(1/2)-1 ~ 0.4142
    idx = pd.to_datetime(["2000-01-01", "2002-01-01", "2004-01-01"])
    lvl = pd.Series([100.0, 200.0, 400.0], index=idx)
    c = st.cagr(lvl)
    assert abs(c - (2 ** 0.5 - 1)) < 0.02


def test_max_drawdown_sign_and_value():
    lvl = pd.Series([100.0, 120.0, 60.0, 90.0])  # peak 120 -> trough 60 = -50%
    assert abs(st.max_drawdown(lvl) - (-0.5)) < 1e-9


def test_beta_recovered_on_pure_linear_relation():
    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0, 0.04, 300))
    stock = 1.3 * mkt                            # exact beta 1.3, no noise
    nw = st.newey_west_alpha_t(stock, mkt, lags=6)
    assert abs(nw["beta"] - 1.3) < 1e-6
    assert abs(nw["alpha_m"]) < 1e-6             # no alpha planted


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
    assert r["mean_excess"] > 0
    assert r["t"] > 2.0


def test_recession_excess_null_when_no_edge():
    n = 120
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(0.0, 0.03, n), index=idx)
    stock = spy + pd.Series(rng.normal(0.0, 0.03, n), index=idx)  # no systematic edge
    r = st.recession_excess_t(stock, spy, _mask_first_half)
    assert abs(r["t"]) < 2.0


# --------------------------------------------------------------------------- #
# NBER window plumbing — pure, no network
# --------------------------------------------------------------------------- #
def test_recession_months_cover_the_three_recessions():
    months = data.recession_months()
    # 2001 (9), GFC (19), COVID (3) = 31 month-starts
    assert len(months) == 31
    assert pd.Timestamp("2008-10-01") in months        # inside the GFC
    assert pd.Timestamp("2020-03-01") in months        # inside COVID
    assert pd.Timestamp("2005-01-01") not in months    # an expansion month


def test_recession_mask_matches_index():
    idx = pd.date_range("2007-06-01", "2008-06-01", freq="MS")
    mask = data.recession_mask(idx)
    # Dec-2007 onward is in the GFC recession; before is not.
    assert not mask[idx.get_loc(pd.Timestamp("2007-06-01"))]
    assert mask[idx.get_loc(pd.Timestamp("2008-01-01"))]
