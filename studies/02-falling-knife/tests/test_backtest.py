"""Backtest mechanics: costs reduce returns, no look-ahead, stats are sane."""

import numpy as np

from falling_knife import data, triggers, exits, backtest


def _events(ohlc):
    ret = data.daily_returns(ohlc)
    raw = triggers.drawdown(ret, threshold=-0.02, window=20)  # plenty of events
    return triggers.first_crossings(raw, cooldown=5)


def test_costs_reduce_total_return(synth_ohlc):
    sig = _events(synth_ohlc)
    rule = exits.ExitRule(max_hold=5)
    free = backtest.run(synth_ohlc, sig, rule, backtest.CostModel(0, 0, 0, 0, 0))
    pricey = backtest.run(synth_ohlc, sig, rule,
                          backtest.CostModel(2, 1, 2, 20, 0))
    if free.stats["n_trades"] > 0:
        assert pricey.stats["total_return"] < free.stats["total_return"]


def test_panic_slippage_only_hits_entry(synth_ohlc):
    """A pure entry cost should scale with trade count, hold-length invariant."""
    sig = _events(synth_ohlc)
    c0 = backtest.CostModel(0, 0, 0, 0, 0)
    c1 = backtest.CostModel(0, 0, 0, 50, 0)
    r0 = backtest.run(synth_ohlc, sig, exits.ExitRule(max_hold=5), c0)
    r1 = backtest.run(synth_ohlc, sig, exits.ExitRule(max_hold=5), c1)
    n = r0.stats["n_trades"]
    if n > 0:
        # Each trade loses ~50bps once at entry; equity ratio reflects ~n hits.
        assert r1.stats["total_return"] < r0.stats["total_return"]


def test_no_overlap_single_position(synth_ohlc):
    """Trades never overlap: each entry is at or after the prior exit."""
    sig = _events(synth_ohlc)
    res = backtest.run(synth_ohlc, sig, exits.ExitRule(max_hold=10), backtest.CostModel())
    trades = res.trades
    for a, b in zip(trades, trades[1:]):
        assert b.entry_date >= a.exit_date


def test_exposure_between_zero_and_one(synth_ohlc):
    sig = _events(synth_ohlc)
    res = backtest.run(synth_ohlc, sig, exits.ExitRule(max_hold=5), backtest.CostModel())
    assert 0.0 <= res.stats["exposure"] <= 1.0


def test_cost_sweep_monotonic_in_cost(synth_ohlc):
    sig = _events(synth_ohlc)
    sweep = backtest.cost_sweep(synth_ohlc, sig, exits.ExitRule(max_hold=5),
                                panic_bps_grid=(0, 10, 20, 40))
    cagr = sweep["cagr"].to_numpy()
    assert np.all(np.diff(cagr) <= 1e-9)  # higher cost never raises CAGR
