"""The synthetic world is deterministic; the inflation-hedge correlation is detected only when gold
loads on inflation; the null shows ~zero correlation; YoY is a 12-month change; crisis-ballast returns
sane fields."""
import numpy as np, pandas as pd
from safe_haven import data, strategy as st


def test_world_deterministic(hedge_world):
    gold, eq, cpi, truth = hedge_world
    gold2, eq2, cpi2, _ = data.synthetic_world(inflation_beta=1.5, seed=69)
    assert np.allclose(gold.to_numpy(), gold2.to_numpy())
    assert np.allclose(cpi.to_numpy(), cpi2.to_numpy())
    assert truth.hedges_inflation


def test_inflation_hedge_detected(hedge_world):
    gold, eq, cpi, _ = hedge_world
    ih = st.inflation_hedge(gold, cpi)
    assert ih["corr"] > 0.3                          # gold tracks inflation when it's built to


def test_null_world_no_hedge(null_world):
    gold, eq, cpi, _ = null_world
    ih = st.inflation_hedge(gold, cpi)
    assert abs(ih["corr"]) < 0.3                      # no inflation link in the null


def test_yoy_is_12m_change():
    s = pd.Series(range(24), index=pd.date_range("2010-01-31", periods=24, freq="ME")) + 100.0
    y = st.yoy(s)
    assert np.isclose(y.iloc[12], (112.0 - 100.0) / 100.0)


def test_crisis_ballast_fields(hedge_world):
    gold, eq, cpi, _ = hedge_world
    cb = st.crisis_ballast(gold.pct_change(), eq)
    assert set(["stock_corr", "gold_in_crash", "eq_in_crash", "gold_up_share", "n_crash"]).issubset(cb)
    assert 0.0 <= cb["gold_up_share"] <= 1.0 or np.isnan(cb["gold_up_share"])
