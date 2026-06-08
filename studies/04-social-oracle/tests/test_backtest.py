"""The follower's backtest: costs, the cost sweep, and capacity."""

import numpy as np

from social_oracle import backtest


def test_costs_make_net_worse_than_gross(panel, events):
    res = backtest.run(panel, events, hold_days=10)
    s = res.stats
    assert s["n_trades"] == len(events)
    assert s["mean_net"] <= s["mean_gross"] + 1e-12   # round-trip cost is a drag


def test_round_trip_frac_scales_with_spread():
    cheap = backtest.CostModel(half_spread_bps=5)
    dear = backtest.CostModel(half_spread_bps=100)
    assert dear.round_trip_frac() > cheap.round_trip_frac()


def test_cost_sweep_monotonic_in_spread(panel, events):
    sweep = backtest.cost_sweep(panel, events, hold_days=10)
    # wider spread -> lower mean net (monotone non-increasing)
    net = sweep["mean_net"].to_numpy()
    assert np.all(np.diff(net) <= 1e-12)


def test_capacity_positive_and_scales_with_edge(panel, events):
    small = backtest.capacity(panel, events, edge_bps=20)
    big = backtest.capacity(panel, events, edge_bps=80)
    assert small["capacity_usd_per_trade"] > 0
    # capacity grows with the square of the edge it has to overcome
    assert big["capacity_usd_per_trade"] > small["capacity_usd_per_trade"]


def test_trades_buy_next_open(panel, events):
    res = backtest.run(panel, events, hold_days=5)
    tr = res.trades[0]
    frame = panel[tr.ticker]
    # entry date is strictly after the mention session (next-open execution)
    assert tr.entry_date in frame.index
    assert tr.holding_days == 5
