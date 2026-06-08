"""The trading rule: mechanics, costs, execution lag, and the rolling driver."""

import numpy as np
import pandas as pd

from pairs_trading import backtest, pairs


def test_cost_model_round_trip():
    cheap = backtest.CostModel(half_spread_bps=2)
    dear = backtest.CostModel(half_spread_bps=40)
    assert dear.leg_cost_frac() > cheap.leg_cost_frac()


def test_signal_opens_on_divergence_closes_on_crossing():
    # a spread that diverges past +2sigma then reverts through zero
    spread = np.array([0.0, 0.5, 1.0, 2.5, 1.5, 0.2, -0.1, -0.3])
    sig = backtest._signal_positions(spread, sigma=1.0, k=2.0)
    assert sig[3] == -1            # opened short-A (spread above +2sigma)
    assert sig[6] == 0             # closed once the spread crossed <= 0


def test_run_produces_positive_gross_on_twins(panel):
    res = backtest.run(panel, top_n=6, form_len=252, trade_len=126, wait=1)
    s = res.stats
    assert s["n_trades"] > 0
    assert s["committed_monthly_gross"] > 0      # the baked-in reversion is harvested
    assert s["mean_trade_net"] <= s["mean_trade_gross"] + 1e-12


def test_costs_drag_net_below_gross(panel):
    res = backtest.run(panel, top_n=6, costs=backtest.CostModel(half_spread_bps=20))
    assert res.stats["committed_monthly_net"] <= res.stats["committed_monthly_gross"] + 1e-12


def test_cost_sweep_monotonic(panel):
    sweep = backtest.cost_sweep(panel, top_n=6, half_spread_grid=(0, 5, 20, 50))
    net = sweep["committed_monthly_net"].to_numpy()
    assert np.all(np.diff(net) <= 1e-9)          # wider spread never helps


def test_market_neutral_book_has_low_beta(panel):
    from pairs_trading import data, robustness
    res = backtest.run(panel, top_n=6)
    mkt = data.market_return(panel)
    neutral = robustness.market_neutrality(res.daily, mkt)
    assert abs(neutral["beta"]) < 0.3            # dollar-neutral by construction


def test_windows_are_non_overlapping():
    w = list(backtest._windows(n_rows=1000, form_len=252, trade_len=126))
    assert w[0] == (0, 252, 378)
    # next trading window starts where the previous ended
    assert w[1][1] == w[0][2]
