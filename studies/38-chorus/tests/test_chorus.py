"""The synthetic panel is deterministic; each component is individually WEAK on the control; the COMBO
Sharpe exceeds the best single component; the components are low-correlation; the combo is ~flat on the
null; the signals are dollar-neutral and causal; and the cost machinery behaves."""

import numpy as np
import pandas as pd

from chorus import costs, data, extension, signals, strategy


def test_panel_deterministic_and_has_alpha(combo_ret, combo):
    p2, _, _ = data.synthetic_panel(combo_strength=1.0, seed=38)
    assert np.allclose(combo_ret.to_numpy(), p2.to_numpy())            # seeded → reproducible
    assert combo[2].has_alpha


def test_each_component_is_individually_weak(combo_ret):
    """No soloist is a star on the control — each standalone Sharpe is modest."""
    sig = signals.all_signals(combo_ret)
    for nm, w in sig.items():
        sh = strategy.summary(strategy.book_returns(w, combo_ret, cost_bps=0.0))["sharpe"]
        assert 0.0 < sh < 2.5, f"{nm} standalone Sharpe {sh:.2f} not in the weak band"


def test_combo_beats_best_component(combo_ret):
    """The whole > the parts: the equal-weight combo Sharpe exceeds every standalone component."""
    sig = signals.all_signals(combo_ret)
    comp_sh = {nm: strategy.summary(strategy.book_returns(w, combo_ret, cost_bps=0.0))["sharpe"]
               for nm, w in sig.items()}
    w_combo = strategy.combine(sig, combo_ret, scheme="equal")
    combo_sh = strategy.summary(strategy.book_returns(w_combo, combo_ret, cost_bps=0.0))["sharpe"]
    assert combo_sh > max(comp_sh.values()), f"combo {combo_sh:.2f} <= best component {max(comp_sh.values()):.2f}"


def test_components_are_low_correlation(combo_ret):
    sig = signals.all_signals(combo_ret)
    assert abs(strategy.avg_pairwise_corr(sig, combo_ret)) < 0.35       # near-decorrelated by construction


def test_combo_flat_on_null(null_ret):
    sig0 = signals.all_signals(null_ret)
    for scheme in ("equal", "risk_parity"):
        w = strategy.combine(sig0, null_ret, scheme=scheme)
        sh = strategy.summary(strategy.book_returns(w, null_ret, cost_bps=0.0))["sharpe"]
        assert abs(sh) < 0.8, f"{scheme} combo Sharpe {sh:.2f} not flat on the null"


def test_signals_are_dollar_neutral_and_gross1(combo_ret):
    for nm, w in signals.all_signals(combo_ret).items():
        net = w.sum(axis=1).abs()
        assert net.iloc[260:].max() < 1e-9, f"{nm} not dollar-neutral"
        gross = w.abs().sum(axis=1)
        assert abs(gross[gross > 0].mean() - 1.0) < 1e-6, f"{nm} gross not normalised to 1"


def test_signals_are_causal():
    """Every signal is lagged: flipping the last day's returns leaves earlier weights untouched."""
    idx = pd.bdate_range("2010-01-04", periods=400)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((400, 6)) * 0.01, index=idx, columns=list("ABCDEF"))
    r2 = r.copy(); r2.iloc[-1] *= -5
    for fn in (signals.momentum, signals.reversal, signals.low_vol):
        w, w2 = fn(r), fn(r2)
        assert w.iloc[:-1].equals(w2.iloc[:-1]), f"{fn.__name__} is not causal"


def test_combine_is_renormalised_and_neutral(combo_ret):
    sig = signals.all_signals(combo_ret)
    w = strategy.combine(sig, combo_ret, scheme="risk_parity")
    gross = w.abs().sum(axis=1)
    assert abs(gross[gross > 0].mean() - 1.0) < 1e-6
    assert w.sum(axis=1).abs().iloc[300:].max() < 1e-9


def test_breadth_and_cost_machinery(combo_ret):
    sig = signals.all_signals(combo_ret)
    bs = extension.breadth_sweep(sig, combo_ret)
    assert len(bs) == 3 and list(bs.columns) == ["components", "sharpe", "sharpe_ratio_to_1", "sqrt_k"]
    assert bs["sharpe"].iloc[-1] > bs["sharpe"].iloc[0]                 # more breadth ⇒ higher Sharpe
    cs = costs.cost_sweep(sig, combo_ret)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]               # higher cost ⇒ lower Sharpe
    assert costs.breakeven_cost_bps(sig, combo_ret) >= 0.0


def test_scheme_comparison_shape(combo_ret):
    sig = signals.all_signals(combo_ret)
    sc = extension.scheme_comparison(sig, combo_ret)
    assert list(sc.index) == ["equal", "risk_parity"]
    assert "sharpe" in sc.columns
