"""The synthetic factor world is deterministic; beta is estimated correctly; the BAB book profits
gross only when a low-beta premium exists; the long leg is levered above 1; the self-financing ledger
behaves (a cash rate drags a net-long-leverage book, spreads strictly reduce the return). All offline
on the seeded synthetic world."""

import numpy as np
import pandas as pd

from free_lunch import data, strategy as st


def test_world_deterministic(premium_world):
    a, m, truth = premium_world
    a2, m2, _ = data.synthetic_cross_section(low_beta_premium=0.04, seed=43)
    assert np.allclose(a.to_numpy(), a2.to_numpy())
    assert np.allclose(m.to_numpy(), m2.to_numpy())
    assert truth.has_premium


def test_rolling_beta_recovers_known_beta():
    """An asset built as 1.5×market + small noise must show a trailing beta ≈ 1.5."""
    idx = pd.bdate_range("2000-01-03", periods=600)
    rng = np.random.default_rng(0)
    m = pd.Series(rng.standard_normal(600) * 0.01, index=idx)
    a = 1.5 * m + pd.Series(rng.standard_normal(600) * 0.001, index=idx)
    beta = st.rolling_beta(a, m, window=252).dropna()
    assert abs(beta.iloc[-1] - 1.5) < 0.1


def test_premium_world_profits_gross_and_levers_up(premium_world):
    a, m, _ = premium_world
    gross, lev = st.bab_returns(a, m, return_leverage=True)
    assert st.summary(gross)["sharpe"] > 0.3   # a real gross edge when the premium exists
    assert lev > 1.3                            # the low-beta leg is genuinely levered


def test_null_world_has_no_gross_edge(null_world):
    a, m, _ = null_world
    gross = st.bab_returns(a, m)
    assert abs(st.summary(gross)["sharpe"]) < 0.4   # nothing to harvest in the null


def test_cash_rate_drags_a_net_long_leverage_book(premium_world):
    """The book is net-long notional (w_L > w_H), so measuring the legs in excess of a positive cash
    rate must lower every month's return vs the rf = 0 construction — the free-funding subsidy the old
    raw-return comparison silently pocketed."""
    a, m, _ = premium_world
    g0 = st.bab_returns(a, m)
    rf = pd.Series(0.04 / 12, index=g0.index)   # constant 4% cash rate
    g4 = st.bab_returns(a, m, rf_monthly=rf)
    diff = (g0 - g4).dropna()
    assert (diff > 0).all()                     # every month pays rf on the net-long slice


def test_spreads_strictly_reduce_return(premium_world):
    a, m, _ = premium_world
    g = (1 + st.bab_returns(a, m)).prod()
    n1 = (1 + st.bab_returns(a, m, financing_spread_ann=0.01, borrow_fee_ann=0.0025)).prod()
    n2 = (1 + st.bab_returns(a, m, financing_spread_ann=0.025, borrow_fee_ann=0.005)).prod()
    assert g > n1 > n2   # the frictions monotonically eat the edge


def test_excess_sharpe_convention():
    """summary(rf=...) computes Sharpe on r − rf but leaves CAGR/vol on the raw series."""
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(1)
    r = pd.Series(0.008 + 0.02 * rng.standard_normal(120), index=idx)
    rf = pd.Series(0.003, index=idx)
    raw, ex = st.summary(r), st.summary(r, rf=rf)
    assert ex["sharpe"] < raw["sharpe"]                 # cash hurdle lowers the Sharpe
    assert np.isclose(ex["cagr"], raw["cagr"])          # CAGR still describes the raw series
