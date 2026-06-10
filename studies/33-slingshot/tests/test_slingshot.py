"""The synthetic panel is deterministic; the dollar-neutral reversal book recovers the cross-sectional
overshoot premium gross and finds none in the null; the book is market-neutral and causal; turnover is
high; and the cost / break-even / holding-period machinery behaves."""

import numpy as np
import pandas as pd

from slingshot import costs, data, extension, strategy


def test_panel_deterministic_and_has_reversion(revert_ret, revert):
    p2, _, _ = data.synthetic_panel(revert_strength=0.06, seed=33)
    assert np.allclose(revert_ret.to_numpy(), p2.to_numpy())          # seeded → reproducible
    assert revert[2].has_reversion


def test_book_recovers_reversion_gross(revert_ret):
    s = strategy.summary(strategy.book_returns(revert_ret, cost_bps=0.0))
    assert s["sharpe"] > 1.0                                          # strong gross on the control


def test_book_flat_on_null(null_ret):
    s = strategy.summary(strategy.book_returns(null_ret, cost_bps=0.0))
    assert abs(s["sharpe"]) < 0.8                                     # nothing to fade in a random walk


def test_weights_are_dollar_neutral(revert_ret):
    w = strategy.reversal_signal(revert_ret, lookback=5)
    net = w.sum(axis=1).abs()
    assert net.iloc[10:].max() < 1e-9                                 # long and short legs net to zero
    gross = w.abs().sum(axis=1)
    assert abs(gross[gross > 0].mean() - 1.0) < 1e-6                  # gross normalised to 1


def test_signal_is_causal():
    """reversal_signal is lagged: flipping the last day's returns leaves earlier weights untouched."""
    idx = pd.bdate_range("2010-01-04", periods=300)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((300, 6)) * 0.01, index=idx, columns=list("ABCDEF"))
    w = strategy.reversal_signal(r, lookback=5)
    r2 = r.copy(); r2.iloc[-1] *= -5
    assert w.iloc[:-1].equals(strategy.reversal_signal(r2, lookback=5).iloc[:-1])


def test_turnover_high_and_breakeven_finite(revert_ret):
    assert strategy.turnover(revert_ret) > 0.1
    be = costs.breakeven_cost_bps(revert_ret)
    assert be >= 0.0
    cs = costs.cost_sweep(revert_ret)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]            # higher cost ⇒ lower Sharpe


def test_holding_and_lookback_sweeps(revert_ret):
    hp = extension.holding_period_sweep(revert_ret, holds=[1, 5, 21])
    assert list(hp.columns) == ["gross_sharpe", "net_sharpe", "turnover_per_day"]
    assert hp["turnover_per_day"].loc[21] < hp["turnover_per_day"].loc[1]
    lb = extension.lookback_sweep(revert_ret, lookbacks=[1, 5, 21])
    assert "sharpe" in lb.columns and len(lb) == 3
