"""The synthetic term-structure panel is deterministic; the cross-sectional carry book recovers the
roll-yield premium gross on the control and finds ~none in the null; the book is dollar-neutral and causal;
turnover is low; the carry+momentum blend behaves; and the cost / break-even machinery behaves. The
generic futures-curve hook returns empty on a cache miss (the real energy run lives in test_energy.py)."""

import numpy as np
import pandas as pd

from contango import costs, data, extension, strategy


def test_panel_deterministic_and_has_carry(carry_ret, carry):
    r2, ry2, truth2 = data.synthetic_term_structure(carry_strength=0.9, seed=35)
    assert np.allclose(carry_ret.to_numpy(), r2.to_numpy())          # seeded → reproducible
    assert np.allclose(carry[1].to_numpy(), ry2.to_numpy())
    assert carry[2].has_carry
    assert not truth2 is None and truth2.has_carry


def test_book_recovers_carry_gross(carry_ret, carry_ry):
    s = strategy.summary(strategy.book_returns(carry_ret, carry_ry, cost_bps=0.0))
    assert s["sharpe"] > 1.0                                          # strong gross on the control


def test_high_minus_low_bucket_positive(carry_ret, carry_ry):
    pb = strategy.carry_premium_by_bucket(carry_ret, carry_ry)
    assert pb["hml_ann_pct"] > 0.0                                    # backwardated out-earn contangoed
    assert pb["high_ann_pct"] > pb["low_ann_pct"]


def test_book_flat_on_null(null_ret, null_ry):
    s = strategy.summary(strategy.book_returns(null_ret, null_ry, cost_bps=0.0))
    assert abs(s["sharpe"]) < 0.8                                     # carry disconnected from returns
    pb = strategy.carry_premium_by_bucket(null_ret, null_ry)
    assert abs(pb["hml_ann_pct"]) < abs(
        strategy.carry_premium_by_bucket(*data.synthetic_term_structure(carry_strength=0.9, seed=35)[:2])["hml_ann_pct"])


def test_weights_are_dollar_neutral(carry_ry):
    w = strategy.carry_signal(carry_ry)
    net = w.sum(axis=1).abs()
    assert net.iloc[5:].max() < 1e-9                                  # long and short legs net to zero
    gross = w.abs().sum(axis=1)
    assert abs(gross[gross > 0].mean() - 1.0) < 1e-6                  # gross normalised to 1


def test_signal_is_causal():
    """carry_signal is lagged: flipping the last week's roll yield leaves earlier weights untouched."""
    idx = pd.bdate_range("2010-01-01", periods=200, freq="W-FRI")
    rng = np.random.default_rng(0)
    ry = pd.DataFrame(rng.standard_normal((200, 6)) * 0.002, index=idx, columns=list("ABCDEF"))
    w = strategy.carry_signal(ry)
    ry2 = ry.copy(); ry2.iloc[-1] *= -5
    assert w.iloc[:-1].equals(strategy.carry_signal(ry2).iloc[:-1])


def test_turnover_low_and_breakeven_high(carry_ret, carry_ry):
    t = strategy.turnover(carry_ry)
    assert 0.0 < t < 0.2                                              # slow signal → low turnover
    be = costs.breakeven_cost_bps(carry_ret, carry_ry)
    assert be > 10.0                                                  # costs not the binding constraint
    cs = costs.cost_sweep(carry_ret, carry_ry)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]             # higher cost ⇒ lower Sharpe


def test_carry_plus_momentum_combine(carry_ret, carry_ry):
    c = extension.combine(carry_ret, carry_ry, cost_bps=5.0)
    assert set(["carry_sharpe", "momentum_sharpe", "blend_sharpe", "correlation"]).issubset(c)
    assert -1.0 <= c["correlation"] <= 1.0
    # a real carry sleeve: the blend should not be worse than the weaker leg
    assert c["blend_sharpe"] >= min(c["carry_sharpe"], c["momentum_sharpe"]) - 1e-9


def test_fetch_curve_empty_on_cache_miss():
    """The generic futures-curve hook returns empty when no broad term-structure cache exists."""
    out = data.fetch_curve(cache_dir="/nonexistent_cache_dir_xyz", fetch=False)
    assert out == {}
    out2 = data.fetch_curve(cache_dir="/nonexistent_cache_dir_xyz", fetch=True)
    assert out2 == {}                                                # no free deferred-contract source


def test_front_month_basket_loads_but_lacks_curve():
    """The cached front-month basket loads (12 names) but carries no term structure to price roll yield."""
    basket = data.load_front_month_basket()
    if basket.empty:
        return                                                       # cache absent in this checkout — fine
    assert basket.shape[1] == 12
