"""The synthetic FX panel is deterministic; the carry book captures the premium gross on the control and
~0 on the full-UIRP null; carry carries the crash signature (negative skew); the carry⊕momentum combo
beats either leg and the legs decorrelate; the signals are causal; the cost machinery behaves. FAST."""

import numpy as np
import pandas as pd

from greenback import costs, data, extension, strategy


def test_panel_deterministic_and_has_carry(fx_xr, fx):
    xr2, _, truth2 = data.synthetic_fx(carry_strength=0.9, trend_strength=0.35, seed=36)
    assert np.allclose(fx_xr.to_numpy(), xr2.to_numpy())              # seeded → reproducible
    assert fx[2].has_carry and fx[2].has_trend


def test_carry_book_captures_premium_gross(fx_xr, fx_rd):
    pb = strategy.carry_premium_by_bucket(fx_xr, fx_rd)
    assert pb["hml_ann_pct"] > 0 and pb["premium_present"]            # high-rate buckets out-earn
    s = strategy.summary(strategy.carry_returns(fx_xr, fx_rd, cost_bps=0.0, target_vol_ann=None))
    assert s["sharpe"] > 0.4                                          # a real premium gross on the control


def test_carry_flat_on_null(null_xr, null_rd):
    pb = strategy.carry_premium_by_bucket(null_xr, null_rd)
    s = strategy.summary(strategy.carry_returns(null_xr, null_rd, cost_bps=0.0, target_vol_ann=None))
    assert abs(pb["hml_ann_pct"]) < 1.5                              # full UIRP ⇒ flat bucket profile
    assert abs(s["sharpe"]) < 0.6                                    # ~0 premium under the null


def test_carry_has_negative_skew_crash(fx_xr, fx_rd):
    """The steamroller: the carry book is negatively skewed on the control (crash signature)."""
    c = strategy.carry_returns(fx_xr, fx_rd, cost_bps=0.0, target_vol_ann=None)
    assert c.skew() < 0.0


def test_combo_beats_legs_and_legs_decorrelate(fx_xr, fx_rd):
    d = extension.diversification(fx_xr, fx_rd, cost_bps=10.0)
    assert d["combo_beats_legs"]                                     # the combo Sharpe > either standalone
    assert d["leg_correlation"] < 0.5                               # carry & momentum pay at different times


def test_momentum_book_has_edge(fx_xr):
    s = strategy.summary(strategy.momentum_returns(fx_xr, cost_bps=0.0, target_vol_ann=None))
    assert s["sharpe"] > 0.2                                         # the trend gives momentum something to ride


def test_carry_weights_dollar_neutral(fx_rd):
    w = strategy.carry_weights(fx_rd).dropna()
    assert w.sum(axis=1).abs().max() < 1e-9                          # long & short legs net to zero
    gross = w.abs().sum(axis=1)
    assert abs(gross[gross > 0].mean() - 1.0) < 1e-6                 # gross exposure normalised to 1


def test_signals_are_causal():
    """carry_weights and momentum_weights are lagged: flipping the last month leaves earlier weights intact."""
    idx = pd.period_range("2000-01", periods=120, freq="M").to_timestamp(how="end")
    rng = np.random.default_rng(0)
    xr = pd.DataFrame(rng.standard_normal((120, 6)) * 0.02, index=idx, columns=[f"C{i}" for i in range(6)])
    rd = pd.DataFrame(np.tile(np.linspace(-0.01, 0.01, 6), (120, 1)), index=idx, columns=xr.columns)
    for fn, arg in [(strategy.momentum_weights, xr), (strategy.carry_weights, rd)]:
        w = fn(arg)
        arg2 = arg.copy(); arg2.iloc[-1] *= -5
        assert w.iloc[:-1].equals(fn(arg2).iloc[:-1])


def test_cost_behaviour_and_breakeven(fx_xr, fx_rd):
    cs = costs.cost_sweep(fx_xr, fx_rd, which="combo")
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]            # higher cost ⇒ lower Sharpe
    be = costs.breakeven_cost_bps(fx_xr, fx_rd, which="carry")
    assert be > 0.0                                                 # carry makes money gross ⇒ positive break-even


def test_dollar_carry_runs(fx_xr, fx_rd):
    s = strategy.summary(strategy.dollar_carry_returns(fx_xr, fx_rd))
    assert np.isfinite(s["sharpe"])                                 # the dollar-carry tilt produces a stream
