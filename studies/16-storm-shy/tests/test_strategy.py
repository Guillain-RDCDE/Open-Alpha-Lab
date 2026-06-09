"""The overlay uses only the past, respects its leverage cap, and on the clustered tape it lifts the
Sharpe at matched risk — while on the flat null it can't, and the gain survives realistic costs."""

import numpy as np
import pandas as pd

from storm_shy import strategy


def test_weights_are_past_only(regime_returns):
    """The weight on day t must use returns only up to t-1 (no look-ahead): it is the lagged
    forecast, so w_t is independent of r_t given the history."""
    w = strategy.managed_weights(regime_returns, target_vol_ann=0.12, window=21)
    # reconstruct the unlagged forecast; the used weight must equal its one-bar lag
    from storm_shy.vol import realized_vol
    target_daily = 0.12 / np.sqrt(strategy.TRADING_DAYS_PER_YEAR)
    unlagged = (target_daily / realized_vol(regime_returns, 21)).clip(upper=2.0)
    assert w.dropna().equals(unlagged.shift(1).reindex(w.index).dropna())


def test_leverage_cap_binds(regime_returns):
    w = strategy.managed_weights(regime_returns, max_leverage=2.0)
    assert w.dropna().max() <= 2.0 + 1e-12


def test_managed_beats_buy_hold_with_regime(regime_returns):
    """Re-timing exposure lifts the (scale-invariant) Sharpe on a clustered tape."""
    cmp = strategy.compare(regime_returns, cost_bps=1.0)
    assert cmp["sharpe_gain_net"] > 0.3
    assert cmp["managed_net"]["sharpe"] > cmp["buy_hold"]["sharpe"]


def test_no_free_gain_when_flat(flat_returns):
    """No regime -> scaling by a noisy 1/sigma adds no systematic Sharpe."""
    cmp = strategy.compare(flat_returns, cost_bps=1.0)
    assert abs(cmp["sharpe_gain_net"]) < 0.25


def test_turnover_is_low_and_cost_sweep_monotone(regime_returns):
    cmp = strategy.compare(regime_returns, cost_bps=1.0)
    assert cmp["turnover_ann"] < 20.0                       # slow forecast -> modest turnover
    sweep = strategy.cost_sweep(regime_returns)
    gains = sweep["sharpe_gain"].to_numpy()
    assert (np.diff(gains) <= 1e-9).all()                   # gain only erodes as cost rises
    assert gains[0] > gains[-1]
