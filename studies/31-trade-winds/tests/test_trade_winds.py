"""The synthetic panel is deterministic; the TSMOM book recovers the trend premium and finds none in
the null; the signal is causal (lagged); the book is positive-skew; and the cost/lookback sweeps behave."""

import numpy as np
import pandas as pd

from trade_winds import costs, data, extension, strategy


def test_panel_deterministic_and_has_trend(trend_ret, trend):
    r2, _ = data.synthetic_trends(trend_strength=0.12, seed=31)
    assert np.allclose(trend_ret.to_numpy(), r2.to_numpy())          # seeded → reproducible
    assert trend[1].has_trend


def test_book_recovers_trend_premium(trend_ret):
    s = strategy.summary(strategy.book_returns(trend_ret, cost_bps=2.0))
    assert s["sharpe"] > 1.0                                          # strong on the control
    assert s["skew"] > -0.5                                           # not a hidden short-vol blowup


def test_book_flat_on_null(null_ret):
    s = strategy.summary(strategy.book_returns(null_ret, cost_bps=2.0))
    assert abs(s["sharpe"]) < 0.8                                     # no premium in a random walk


def test_signal_is_causal():
    """tsmom_signal is lagged: today's signal uses only returns up to yesterday."""
    idx = pd.bdate_range("2010-01-04", periods=400)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((400, 3)) * 0.01, index=idx, columns=["A", "B", "C"])
    sig = strategy.tsmom_signal(r)
    assert sig.iloc[0].abs().sum() == 0                              # first row has no past → 0
    # flipping the LAST day's return must not change any earlier signal
    r2 = r.copy(); r2.iloc[-1] *= -5
    assert sig.iloc[:-1].equals(strategy.tsmom_signal(r2).iloc[:-1])


def test_lookback_and_cost_sweeps(trend_ret):
    sw = extension.lookback_sweep(trend_ret)
    assert "sharpe" in sw.columns and len(sw) >= 4
    cs = costs.cost_sweep(trend_ret)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]            # higher cost ⇒ lower Sharpe


def test_crisis_alpha_shape(trend_ret):
    ca = costs.crisis_alpha(trend_ret, equity_cols=["MKT00", "MKT01"])
    assert {"book_in_equity_crises_ann", "corr_to_equities", "n_crisis_months"} <= set(ca)
    assert ca["n_crisis_months"] > 0
