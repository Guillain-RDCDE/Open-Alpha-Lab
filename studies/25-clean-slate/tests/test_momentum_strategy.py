"""The residualiser is causal, the residual-momentum engine recovers the baked premium (and finds little
on the null), and the residual-WML book makes money with momentum and not without it."""

import numpy as np

from clean_slate import data, momentum, strategy


def test_deterministic_and_momentum(mom):
    panel, _, truth = mom
    panel2, _, _ = data.synthetic_panel(mom_strength=0.0016, seed=25)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())
    assert truth.has_momentum


def test_residual_is_causal_and_shaped(mom_panel, mom_market):
    rr = momentum.residual_returns(mom_panel, mom_market, beta_window=252)
    assert rr.shape == mom_panel.shape
    # the residual strips most of the market: its cross-sectional mean is far less market-correlated
    res_mkt_corr = rr.mean(axis=1).corr(mom_market)
    raw_mkt_corr = mom_panel.mean(axis=1).corr(mom_market)
    assert abs(res_mkt_corr) < abs(raw_mkt_corr)


def test_spread_recovers_momentum(mom_panel, mom_market):
    sp = momentum.momentum_spread(mom_panel, mom_market, residual=True)
    assert sp["wml_ann_pct"] > 5.0
    assert sp["momentum_present"]


def test_spread_small_on_null(null_panel, null_market):
    sp = momentum.momentum_spread(null_panel, null_market, residual=True)
    assert abs(sp["wml_ann_pct"]) < 12.0       # no persistent residual drift -> small (noisy) spread


def test_wml_beats_market_on_momentum(mom_panel, mom_market):
    cmp = strategy.compare(mom_panel, mom_market, cost_bps=5.0)
    assert cmp["residual"]["sharpe"] > 0.8
    assert cmp["turnover_ann"] < 20.0


def test_wml_no_edge_on_null(null_panel, null_market):
    cmp = strategy.compare(null_panel, null_market, cost_bps=5.0)
    assert abs(cmp["residual"]["sharpe"]) < 0.7
