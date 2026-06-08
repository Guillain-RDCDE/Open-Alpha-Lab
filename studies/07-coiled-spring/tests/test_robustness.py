"""The gauntlet must compute the right shapes and the right sign of the synthetic edge."""

import numpy as np
import pandas as pd

from coiled_spring import backtest, data, robustness
from coiled_spring.signals import Params


def test_signal_vs_baseline_shape(synthetic):
    frames, _ = synthetic
    sb = robustness.signal_vs_baseline(frames, horizons=(5, 10))
    assert list(sb.index) == [5, 10]
    for col in ["n_signals", "mean_breakout", "mean_excess", "hac_t"]:
        assert col in sb.columns


def test_synthetic_short_horizon_excess_is_positive(synthetic):
    """The planted run lives in the first ~10 bars: excess must be clearly positive there."""
    frames, _ = synthetic
    sb = robustness.signal_vs_baseline(frames, horizons=(10,))
    assert sb.loc[10, "mean_excess"] > 0.0
    assert sb.loc[10, "hac_t"] > 2.0


def test_cost_sweep_is_monotone_decreasing(synthetic):
    frames, _ = synthetic
    sweep = robustness.cost_sweep(frames, cost_grid_bps=(0, 10, 50))
    net = sweep["mean_net"].to_numpy()
    assert net[0] >= net[1] >= net[2]          # more cost can only lower net return


def test_decay_by_year_sums_to_total(synthetic):
    frames, _ = synthetic
    led = backtest.run_universe(frames, cost_bps=15.0)
    decay = robustness.decay_by_year(frames, cost_bps=15.0)
    assert decay["n_trades"].sum() == len(led)


def test_capacity_handles_empty():
    out = robustness.capacity(pd.Series(dtype=float), pd.DataFrame())
    assert np.isnan(out["median_dollar_vol"])
