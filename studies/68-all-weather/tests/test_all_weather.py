"""The synthetic world is deterministic; inverse-vol weights are normalised and tilt to low-vol assets;
risk parity beats equal-weight on Sharpe when vols differ and ties it when they don't; the backtest is
a valid return series."""
import numpy as np, pandas as pd
from all_weather import data, strategy as st


def test_world_deterministic(spread_world):
    ret, truth = spread_world
    ret2, _ = data.synthetic_world(vol_spread=0.012, seed=68)
    assert np.allclose(ret.to_numpy(), ret2.to_numpy())
    assert truth.vols_differ


def test_weights_normalised_and_tilted():
    vol = pd.Series({"SPY": 0.02, "IEF": 0.005, "GLD": 0.01, "DBC": 0.015})
    w = st.inverse_vol_weights(vol)
    assert np.isclose(w.sum(), 1.0)
    assert w["IEF"] > w["SPY"]                       # lowest-vol asset gets the most weight


def test_risk_parity_helps_when_vols_differ(spread_world):
    ret, _ = spread_world
    rp = st.stats(st.backtest(ret, st.risk_parity))
    ew = st.stats(st.backtest(ret, st.equal_weight))
    assert rp["sharpe"] > ew["sharpe"]
    assert rp["vol"] < ew["vol"]


def test_no_edge_when_vols_equal(flat_world):
    ret, _ = flat_world
    rp = st.stats(st.backtest(ret, st.risk_parity))
    ew = st.stats(st.backtest(ret, st.equal_weight))
    assert abs(rp["sharpe"] - ew["sharpe"]) < 0.25   # risk parity ≈ equal weight when vols match


def test_backtest_is_a_return_series(spread_world):
    ret, _ = spread_world
    r = st.backtest(ret, st.risk_parity)
    assert len(r) == len(ret)
    assert r.notna().all()
