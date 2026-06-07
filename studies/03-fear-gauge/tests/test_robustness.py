"""Block bootstrap, window-selection, the martingale ruin sim, and deflation."""

import numpy as np

from fear_gauge import robustness, triggers


def test_block_bootstrap_returns_interval(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.level(synth_gauge, 30), cooldown=21)
    out = robustness.block_bootstrap_excess(synth_market, sig, horizon=21, n_iter=200)
    assert out["ci_low"] <= out["mean"] <= out["ci_high"]
    assert 0.0 <= out["p_excess_le_0"] <= 1.0


def test_window_sensitivity_rows(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.spike(synth_gauge), cooldown=21)
    tbl = robustness.window_sensitivity(synth_market, sig, horizon=21)
    assert "full history" in tbl.index
    assert {"mean_cond", "mean_uncond", "excess"}.issubset(tbl.columns)


def test_martingale_ruin_fields(synth_market, synth_gauge):
    out = robustness.martingale_ruin(synth_market, synth_gauge, rung1=30, rung2=50)
    for k in ("n_episodes", "p_ruin", "worst_drawdown", "worst_terminal", "mean_terminal"):
        assert k in out
    if out["n_episodes"] > 0:
        assert 0.0 <= out["p_ruin"] <= 1.0
        assert out["worst_drawdown"] <= 0.0


def test_deflated_sharpe_monotonic_in_trials():
    # More trials => a high Sharpe is less impressive (lower deflated probability).
    d1 = robustness.deflated_sharpe(2.0, n_trials=1, n_obs=500)
    d50 = robustness.deflated_sharpe(2.0, n_trials=50, n_obs=500)
    assert d1 >= d50
    assert 0.0 <= d50 <= 1.0


def test_split_sample_partitions(synth_market):
    a, b = robustness.split_sample(synth_market, frac=0.6)
    assert len(a) + len(b) == len(synth_market)
    assert a.index.max() < b.index.min()
