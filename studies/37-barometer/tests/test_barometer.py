"""The synthetic macro world is deterministic; the macro-momentum book captures the premium gross on the
control and earns ~0 on the null; the books are dollar-neutral and causal; the inflation-hedge tilt pays
more in rising- than falling-inflation regimes; turnover is low and the cost machinery behaves."""

import numpy as np
import pandas as pd
import pytest

from barometer import costs, data, extension, strategy


def test_synthetic_deterministic_and_has_macro(control):
    r, m, truth = control
    r2, m2, _ = data.synthetic_macro(macro_strength=1.0, seed=37)
    assert np.allclose(r.to_numpy(), r2.to_numpy())                  # seeded → reproducible
    assert np.allclose(m.to_numpy(), m2.to_numpy())
    assert truth.has_macro
    assert list(r.columns) == data.ASSET_ORDER
    assert list(m.columns) == ["growth", "inflation"]


def test_macro_momentum_captures_premium_gross(control):
    r, m, _ = control
    s = strategy.summary(strategy.book_returns(r, m, kind="macro_momentum", cost_bps=0.0))
    assert s["sharpe"] > 0.7                                         # strong gross premium on the control


def test_books_flat_on_null(null):
    r0, m0, _ = null
    for kind in ("macro_momentum", "inflation"):
        s = strategy.summary(strategy.book_returns(r0, m0, kind=kind, cost_bps=0.0))
        assert abs(s["sharpe"]) < 0.5                                # nothing to predict in pure noise


def test_weights_dollar_neutral_and_gross_one(control):
    _, m, _ = control
    for wfn in (strategy.macro_momentum_weights, strategy.inflation_tilt_weights):
        w = wfn(m)
        net = w.sum(axis=1).abs()
        assert net.iloc[12:].max() < 1e-9                            # long and short legs net to zero
        gross = w.abs().sum(axis=1)
        assert abs(gross[gross > 0].mean() - 1.0) < 1e-6            # gross normalised to 1


def test_weights_are_causal(control):
    """Weights are lagged: changing the LAST month's macro state leaves earlier weights untouched."""
    _, m, _ = control
    w = strategy.macro_momentum_weights(m)
    m2 = m.copy()
    m2.iloc[-1] = m2.iloc[-1] * 5.0 + 1.0
    w2 = strategy.macro_momentum_weights(m2)
    assert np.allclose(w.iloc[:-1].to_numpy(), w2.iloc[:-1].to_numpy())


def test_inflation_tilt_pays_more_in_rising_inflation(control):
    r, m, _ = control
    sp = extension.regime_split(r, m, kind="inflation", cost_bps=0.0)
    assert sp.loc["rising", "ann_return_pct"] > sp.loc["falling", "ann_return_pct"]
    assert sp.loc["rising", "sharpe"] > sp.loc["falling", "sharpe"]


def test_turnover_low_and_breakeven_high(control):
    r, m, _ = control
    # macro signals are slow → low turnover and a comfortably high break-even (cost is not the threat)
    assert strategy.turnover_ann(m, kind="macro_momentum") < 12.0
    be = costs.breakeven_cost_bps(r, m, kind="macro_momentum")
    assert be > 20.0
    cs = costs.cost_sweep(r, m, kind="macro_momentum")
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]            # higher cost ⇒ lower Sharpe


def test_fetch_macro_cache_miss_returns_empty():
    """Offline cache-miss path returns {} — never blocks the verdict."""
    assert data.fetch_macro(cache_dir="/nonexistent_cache_dir_xyz", fetch=False) == {}


def test_real_book_runs_on_cache_if_present():
    """Cache-gated real smoke test: if the cached parquets exist, the real ETF book runs end-to-end,
    is aligned/lagged to the post-2007 sample, and both books and the regime split produce finite stats."""
    bundle = data.fetch_macro()
    if not bundle:
        pytest.skip("real macro/ETF cache not present in this environment")
    macro, rets = bundle["macro"], bundle["assets"]
    assert list(rets.columns) == data.REAL_ASSET_ORDER
    assert {"growth", "inflation"}.issubset(macro.columns)
    assert macro.index.equals(rets.index) and len(rets) > 100   # ~18y post-2007 monthly sample
    spec = dict(assets=data.REAL_ASSETS, order=data.REAL_ASSET_ORDER)
    for kind in ("macro_momentum", "inflation"):
        s = strategy.summary(strategy.book_returns(rets, macro, kind=kind, cost_bps=5.0, **spec))
        assert np.isfinite(s["sharpe"])
    sp = extension.regime_split(rets, macro, kind="inflation", cost_bps=5.0, **spec)
    assert {"rising", "falling"}.issubset(sp.index)
