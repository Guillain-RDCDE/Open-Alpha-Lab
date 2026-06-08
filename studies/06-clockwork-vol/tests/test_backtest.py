"""Walk-forward forecast skill — must beat the random-phase null on a *real* cycle."""

import numpy as np
import pandas as pd

from vix_cycles import backtest, data


def test_dominant_period_finds_injected(synth):
    series, injected = synth
    P = backtest.dominant_period(series, band=(20.0, 200.0))
    assert min(abs(P - c.period) for c in injected) <= 0.2 * 80.0


def test_direction_skill_beats_null_on_real_cycle(synth):
    series, _ = synth
    r = backtest.oos_direction_skill(series, band=(20.0, 200.0), horizon=20,
                                     min_train=750, step=10, n_null=200, seed=0)
    assert r["skill"] > 0.5
    assert r["p_value"] < 0.10        # the learned phase forecasts better than scrambled


def test_direction_skill_no_edge_on_red_noise(red_noise):
    r = backtest.oos_direction_skill(red_noise, band=(20.0, 200.0), horizon=20,
                                     min_train=750, step=10, n_null=200, seed=0)
    assert r["p_value"] > 0.05        # no learnable phase in structureless noise


def test_phase_trade_runs_and_is_causal(synth):
    series, _ = synth
    # A toy 'index' that falls when the synthetic vol cycle is high → the trade should help.
    px = pd.Series(np.exp(np.cumsum(-0.3 * (series.values - series.values.mean()) * 0.01)),
                   index=series.index, name="px")
    res = backtest.phase_trade(series, px, band=(20.0, 200.0), horizon=20,
                               min_train=750, refit=5, n_null=50, seed=0)
    assert res.daily.iloc[:750].abs().sum() == 0.0    # no position before min_train: causal
    assert "sharpe_net" in res.stats and "p_value_vs_null" in res.stats
