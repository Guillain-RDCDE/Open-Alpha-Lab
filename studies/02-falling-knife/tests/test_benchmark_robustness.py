"""The decisive benchmark and the robustness tooling behave as designed."""

import numpy as np
import pandas as pd

from falling_knife import data, triggers, exits, benchmark, robustness


def _market_with_bounce(n=4000, bounce=0.6, seed=7):
    """Synthetic market that hands back part of any drop > 2% (a real edge)."""
    rng = np.random.default_rng(seed)
    vol = 0.012
    r = 0.0003 + vol * rng.standard_normal(n)
    add = np.zeros(n)
    for t in range(n):
        if r[t] < -0.02:
            for k in (1, 2, 3):
                if t + k < n:
                    add[t + k] += -bounce * r[t] / 3.0
    r = r + add
    close = 100 * np.cumprod(1 + r)
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * (1 + 0.003 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.standard_normal(n)))
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_benchmark_detects_injected_edge():
    ohlc = _market_with_bounce(bounce=0.6)
    ret = data.daily_returns(ohlc)
    events = triggers.first_crossings(triggers.close_to_close(ret), cooldown=10)
    bench = benchmark.conditional_vs_unconditional(ohlc, events, horizons=(5,), n_iter=500)
    assert bench.loc[5, "excess"] > 0
    assert bench.loc[5, "p_greater"] < 0.05


def test_benchmark_flat_when_no_edge():
    ohlc = _market_with_bounce(bounce=0.0)  # no edge
    ret = data.daily_returns(ohlc)
    events = triggers.first_crossings(triggers.close_to_close(ret), cooldown=10)
    bench = benchmark.conditional_vs_unconditional(ohlc, events, horizons=(5,), n_iter=500)
    assert bench.loc[5, "p_greater"] > 0.10  # not significantly better than random


def test_benchmark_empty_signal_returns_empty_frame():
    ohlc = _market_with_bounce(bounce=0.0)
    empty = pd.Series(False, index=ohlc.index)
    bench = benchmark.conditional_vs_unconditional(ohlc, empty, horizons=(5,), n_iter=50)
    assert bench.empty


def test_deflated_sharpe_penalises_trials():
    # Same Sharpe is less impressive after many trials.
    few = robustness.deflated_sharpe(1.5, n_trials=1, n_obs=2000)
    many = robustness.deflated_sharpe(1.5, n_trials=200, n_obs=2000)
    assert few > many


def test_block_bootstrap_ci_brackets_mean():
    ohlc = _market_with_bounce(bounce=0.6)
    ret = data.daily_returns(ohlc)
    events = triggers.first_crossings(triggers.close_to_close(ret), cooldown=10)
    bb = robustness.block_bootstrap_excess(ohlc, events, horizon=5, n_iter=500)
    assert bb["ci_low"] <= bb["mean"] <= bb["ci_high"]


def test_oos_best_cell_runs():
    ohlc = _market_with_bounce(bounce=0.6)
    ret = data.daily_returns(ohlc)
    sigs = {n: f(ret) for n, f in triggers.TRIGGERS.items()}
    out = robustness.oos_best_cell(ohlc, sigs, exits.default_grid())
    assert "oos_sharpe" in out and "verdict" in out
