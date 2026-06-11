"""The synthetic market is deterministic; the vol-targeted book cuts drawdown (the real part);
the two financing models coincide at a zero markup and charge each dollar exactly once (no
double-charged risk-free rate); the CFD markup lands on the full notional while the margin
account pays only on the borrowed slice (the house edge); and the levered book never out-earns
buy-and-hold at any markup (the mirage)."""

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


def test_models_coincide_at_zero_markup(price, rate):
    """The keystone identity: margin and futures funding are the SAME at markup = 0 —
    what differs between accounts is only where the broker's markup lands."""
    e = strategy.exposure(price)
    a = costs.net_returns(e, price, rate, mode="margin", markup=0.0)
    b = costs.net_returns(e, price, rate, mode="futures", markup=0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_no_double_charged_riskfree():
    """Exposure ≡ 1, zero markup, no dividends/spread → net return IS the price return.
    (The bug this guards against: financing the full notional while crediting cash interest
    only on the un-deployed fraction, which silently subtracts rf from a fully-invested book.)"""
    idx = pd.bdate_range("2010-01-04", periods=300)
    price = pd.Series(100 * np.cumprod(1 + 0.0003 * np.ones(300)), index=idx, name="price")
    rate = pd.Series(0.04, index=idx, name="short_rate")          # a fat bill rate, to catch any leak
    e = pd.Series(1.0, index=idx)
    ret = price.pct_change().dropna()
    for mode in ("margin", "futures"):
        r = costs.net_returns(e, price, rate, mode=mode, markup=0.0, div_yield=0.0, spread_bps=0.0)
        assert np.allclose(r.to_numpy(), ret.to_numpy()), mode


def test_flat_book_earns_the_bill():
    """Exposure ≡ 0 → the account is all cash and earns exactly the bill rate, both models."""
    idx = pd.bdate_range("2010-01-04", periods=300)
    price = pd.Series(np.linspace(100, 120, 300), index=idx, name="price")
    rate = pd.Series(0.04, index=idx, name="short_rate")
    e = pd.Series(0.0, index=idx)
    for mode in ("margin", "futures"):
        r = costs.net_returns(e, price, rate, mode=mode, div_yield=0.0, spread_bps=0.0)
        assert np.allclose(r.to_numpy(), 0.04 / costs.TRADING_DAYS), mode


def test_house_edge_is_the_markup_on_the_full_notional(price, rate):
    """The CFD pays the markup on the whole notional, the margin account only on the borrowed
    slice — so the CFD's bill is the bigger one, and ≈ avg_exposure × markup per year."""
    e = strategy.exposure(price)
    he = costs.house_edge(e, price, rate, markup=0.025)
    assert he["house_edge_ann"] > he["margin_edge_ann"] >= 0.0
    expected = he["avg_exposure"] * 0.025                          # the back-of-envelope identity
    assert 0.6 * expected < he["house_edge_ann"] < 1.4 * expected


def test_levered_book_does_not_beat_buy_and_hold(price, rate):
    e = strategy.exposure(price)
    marg = strategy.summary(costs.net_returns(e, price, rate, mode="margin"), rf=rate)
    bh = strategy.summary(costs.buy_and_hold(price), rf=rate)
    assert marg["cagr"] < bh["cagr"]                               # the mirage: no return edge
    sweep = extension.financing_sweep(price, rate)
    assert (sweep["margin_edge_cagr"] < 0).all()                   # negative at every markup…
    assert (sweep["cfd_edge_cagr"] <= sweep["margin_edge_cagr"] + 1e-12).all()   # …and worse on a CFD
    assert sweep["cfd_edge_cagr"].is_monotonic_decreasing          # the markup is the dial


def test_excess_sharpe_convention(price, rate):
    """summary(rf=...) computes the Sharpe on returns in excess of the bill — strictly lower
    than the raw-return Sharpe whenever rates are positive — while CAGR stays total-return."""
    bh = costs.buy_and_hold(price)
    raw, ex = strategy.summary(bh), strategy.summary(bh, rf=rate)
    assert ex["sharpe"] < raw["sharpe"]
    assert ex["cagr"] == raw["cagr"]
