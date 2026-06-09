"""The low-vol long-only book beats the market on the anomaly tape, the short-leg borrow fee only ever
erodes the long-short, and on the null the defensive tilt vanishes."""

import numpy as np

from dull_roar import strategy


def test_summary_keys(anomaly_panel, anomaly_market):
    s = strategy.summary(anomaly_market)
    for k in ("sharpe", "ann_return", "vol_ann", "cagr", "max_drawdown", "n_days"):
        assert k in s


def test_low_vol_beats_market_with_anomaly(anomaly_panel, anomaly_market):
    cmp = strategy.compare(anomaly_panel, market=anomaly_market, cost_bps=1.0)
    assert cmp["low_minus_market_sharpe"] > 0.1            # the defensive tilt shows up
    assert cmp["turnover_low"] < 5.0                       # vol deciles are sticky


def test_low_vol_no_edge_on_null(null_panel, null_market):
    cmp = strategy.compare(null_panel, market=null_market, cost_bps=1.0)
    assert abs(cmp["low_minus_market_sharpe"]) < 0.15      # fair CAPM: no risk-adjusted edge to find


def test_borrow_sweep_monotone(anomaly_panel, anomaly_market):
    sw = strategy.borrow_sweep(anomaly_panel, market=anomaly_market)
    s = sw["long_short_sharpe"].to_numpy()
    assert (np.diff(s) <= 1e-9).all()                     # a higher borrow fee can only hurt
    assert s[0] > s[-1]


def test_cost_sweep_monotone(anomaly_panel, anomaly_market):
    sw = strategy.cost_sweep(anomaly_panel, market=anomaly_market)
    g = sw["low_minus_market"].to_numpy()
    assert (np.diff(g) <= 1e-9).all()
