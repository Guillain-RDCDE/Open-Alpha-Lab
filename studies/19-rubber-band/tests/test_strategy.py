"""The timing book uses only the past, churns ~daily, makes money on the reversal tape and nothing on
the null, and its edge erodes monotonically with cost."""

import numpy as np

from rubber_band import strategy, ibs


def test_weights_are_past_only(reversal_ohlc):
    """The weight on day t is set from yesterday's IBS (shift(1)) — no look-ahead."""
    w = strategy.timing_weights(reversal_ohlc)
    raw = (1.0 - 2.0 * ibs.ibs(reversal_ohlc))
    assert w.dropna().equals(raw.shift(1).reindex(w.index).dropna())


def test_turnover_is_huge(reversal_ohlc):
    """A one-day signal churns the whole book daily -> very high annualised turnover."""
    assert strategy.turnover_ann(reversal_ohlc) > 100.0


def test_makes_money_on_reversal(reversal_ohlc):
    s = strategy.summary(strategy.timing_returns(reversal_ohlc, cost_bps=0.0))
    assert s["sharpe"] > 0.8


def test_no_edge_on_null(null_ohlc):
    s = strategy.summary(strategy.timing_returns(null_ohlc, cost_bps=0.0))
    assert abs(s["sharpe"]) < 0.3


def test_cost_sweep_monotone(reversal_ohlc):
    basket = {"A": reversal_ohlc}
    sw = strategy.cost_sweep(basket)
    sr = sw["sharpe"].to_numpy()
    assert (np.diff(sr) <= 1e-9).all()
    assert sr[0] > sr[-1]


def test_cross_sectional_runs():
    """The §4.4 dollar-neutral book builds across a small multi-name basket and returns a daily stream."""
    from rubber_band import data
    basket = {f"A{i}": data.synthetic_ohlc(kappa=0.003, seed=i)[0] for i in range(6)}
    xs = strategy.cross_sectional_returns(basket, decile_frac=0.34)
    assert xs.notna().sum() > 100
