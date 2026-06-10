"""The synthetic market is deterministic; the vol-targeted book cuts drawdown (the real part); honest
full-notional financing costs more than the idealized accounting and never beats buy-and-hold (the
mirage); and the exposure rules behave (trend gate flattens, vol-target caps)."""

import numpy as np
import pandas as pd

from house_edge import costs, data, extension, strategy


def test_market_deterministic_and_sane(price, rate):
    p2, r2, _ = data.synthetic_market(seed=30)
    assert np.allclose(price.to_numpy(), p2.to_numpy())          # seeded → reproducible
    assert np.allclose(rate.to_numpy(), r2.to_numpy())
    bh = strategy.summary(costs.buy_and_hold(price))
    assert 0.02 < bh["cagr"] < 0.20                               # an equity-like control, not a degenerate path
    assert -0.75 < bh["max_drawdown"] < -0.15                     # real bear regimes, not a ruin


def test_exposure_respects_trend_gate_and_cap(price):
    e = strategy.exposure(price, target_vol=0.15, lev_cap=2.0)
    assert e.min() >= 0.0 and e.max() <= 2.0 + 1e-9
    ma = price.rolling(200).mean()
    below = (price < ma).reindex(e.index).fillna(False)
    # deep below the average with no capitulation, exposure is frequently flattened
    assert (e[below] == 0).mean() > 0.3


def test_drawdown_protection_is_real(price, rate):
    dd = extension.drawdown_protection(price, rate)
    assert abs(dd["strat_max_drawdown"]) < abs(dd["bh_max_drawdown"])   # the book cuts drawdown
    assert dd["drawdown_reduction"] > 0.03


def test_honest_costs_more_than_idealized(price, rate):
    e = strategy.exposure(price)
    he = costs.house_edge(e, price, rate)
    assert he["cagr_idealized"] > he["cagr_honest"]                # full-notional financing costs more
    assert he["house_edge_ann"] > 0.0


def test_levered_book_does_not_beat_buy_and_hold(price, rate):
    e = strategy.exposure(price)
    hon = strategy.summary(costs.net_returns(e, price, rate, mode="honest"))
    bh = strategy.summary(costs.buy_and_hold(price))
    assert hon["cagr"] < bh["cagr"]                                # the mirage: no return edge
    sweep = extension.financing_sweep(price, rate)
    assert (sweep["edge_vs_bh_cagr"] < 0).all()                    # negative at every markup


def test_idealized_and_honest_agree_when_unlevered_and_frictionless():
    """With exposure ≡ 1, zero markup, no dividends/cash, both cost models collapse to the price return."""
    idx = pd.bdate_range("2010-01-04", periods=300)
    price = pd.Series(100 * np.cumprod(1 + 0.0003 * np.ones(300)), index=idx, name="price")
    rate = pd.Series(0.0, index=idx, name="short_rate")
    e = pd.Series(1.0, index=idx)
    a = costs.net_returns(e, price, rate, mode="idealized", markup=0.0, div_yield=0.0, spread_bps=0.0)
    b = costs.net_returns(e, price, rate, mode="honest", markup=0.0, div_yield=0.0, spread_bps=0.0,
                          cash_earns=False)
    assert np.allclose(a.to_numpy(), b.to_numpy())
