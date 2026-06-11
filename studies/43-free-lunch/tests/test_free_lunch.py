"""The synthetic factor world is deterministic; beta is estimated correctly; the BAB book profits
gross only when a low-beta premium exists; the long leg is levered above 1; and realistic financing
strictly reduces the return. All offline on the seeded synthetic world."""

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


def test_financing_strictly_reduces_return(premium_world):
    a, m, _ = premium_world
    g = (1 + st.bab_returns(a, m, financing_ann=0.0)).prod()
    n3 = (1 + st.bab_returns(a, m, financing_ann=0.03, borrow_ann=0.005)).prod()
    n5 = (1 + st.bab_returns(a, m, financing_ann=0.05, borrow_ann=0.01)).prod()
    assert g > n3 > n5   # the leverage bill monotonically eats the edge
