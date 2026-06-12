"""The synthetic world is deterministic; a high-beta BTC shows a positive stock correlation and falls in
equity crashes, while a haven BTC does not; gold stays ~uncorrelated with stocks; the sleeve table is
well-formed."""
import numpy as np, pandas as pd
from digital_gold import data, strategy as st


def test_world_deterministic(risk_world):
    ret, truth = risk_world
    ret2, _ = data.synthetic_world(btc_stock_beta=1.5, seed=70)
    assert np.allclose(ret.to_numpy(), ret2.to_numpy())
    assert truth.is_risk_asset


def test_risk_asset_correlates_with_stocks(risk_world):
    ret, _ = risk_world
    c = st.correlations(ret)
    assert c["btc_stock"] > 0.3                        # behaves like a risk asset
    assert abs(c["btc_gold"]) < 0.2                    # not gold-like


def test_haven_world_is_uncorrelated(haven_world):
    ret, _ = haven_world
    c = st.correlations(ret)
    assert abs(c["btc_stock"]) < 0.2                   # the counterfactual digital gold


def test_risk_asset_falls_in_crashes(risk_world):
    ret, _ = risk_world
    m = (1 + ret).resample("ME").prod() - 1
    cb = st.crisis_behavior(m, crash=-0.05)            # gentler threshold ⇒ a robust crash sample
    assert cb["btc_in_crash"] < cb["eq_in_crash"]      # falls *harder* than stocks
    assert cb["btc_up_share"] < 0.5


def test_sleeve_table_wellformed(risk_world):
    ret, _ = risk_world
    sl = st.sleeve_effect(ret["SPY"], ret["BTC-USD"], weights=(0.0, 0.05))
    assert list(sl.index) == ["0% BTC", "5% BTC"]
    assert {"ann", "vol", "sharpe", "max_drawdown"}.issubset(sl.columns)
