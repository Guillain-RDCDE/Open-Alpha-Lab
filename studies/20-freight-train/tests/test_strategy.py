"""The TSMOM book uses only the past, is gross-normalised, turns over slowly, beats the basket on the
trend tape and adds nothing on the null, and its edge barely moves with cost (the low-turnover point)."""

import numpy as np

from trend_follow import strategy


def test_weights_gross_normalised_and_lagged(trend_panel):
    panel, _ = trend_panel
    w = strategy.tsmom_weights(panel)
    gross = w.abs().sum(axis=1).dropna()
    # on days with a full position the gross exposure is ~1 (normalised)
    assert np.allclose(gross[gross > 0].round(6).unique().max(), 1.0, atol=1e-6)


def test_turnover_is_low(trend_panel):
    panel, _ = trend_panel
    assert strategy.turnover_ann(panel) < 12.0      # slow signal -> a few rebalances' worth a year


def test_beats_basket_on_trend(trend_panel):
    panel, _ = trend_panel
    cmp = strategy.compare(panel, cost_bps=2.0)
    assert cmp["tsmom"]["sharpe"] > 0.5
    assert cmp["sharpe_gain"] > 0.3


def test_no_edge_on_null(null_panel):
    panel, _ = null_panel
    cmp = strategy.compare(panel, cost_bps=2.0)
    assert abs(cmp["tsmom"]["sharpe"]) < 0.35


def test_cost_sweep_monotone_and_shallow(trend_panel):
    panel, _ = trend_panel
    sw = strategy.cost_sweep(panel)
    s = sw["sharpe"].to_numpy()
    assert (np.diff(s) <= 1e-9).all()               # cost only hurts
    assert (s[0] - s[-1]) < 0.5                      # but barely -- low turnover, far-off break-even
