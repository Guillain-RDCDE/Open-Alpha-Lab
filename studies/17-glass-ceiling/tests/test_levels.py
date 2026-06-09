"""The geometry: resistance is a trailing high with no look-ahead, the confirmation trigger needs N
closes above it, and the stop honours the 1% floor that stops a too-tight stop manufacturing losses."""

import numpy as np
import pandas as pd

from glass_ceiling import levels


def _bars(close):
    close = np.asarray(close, float)
    return pd.DataFrame({"Open": close, "High": close, "Low": close,
                         "Close": close, "Volume": np.ones_like(close)})


def test_resistance_is_past_only():
    bars = _bars([1, 2, 3, 2, 5])
    res = levels.resistance(bars, lookback=3)
    assert np.isnan(res.iloc[0])               # warm-up
    # at index 3, prior 3 closes are [1,2,3] -> max 3
    assert res.iloc[3] == 3.0
    # at index 4, prior 3 closes are [2,3,2] -> max 3
    assert res.iloc[4] == 3.0


def test_confirm_needs_consecutive_closes_above():
    # one close above then back below -> no 2-bar trigger; two above -> trigger
    bars = _bars([10, 10, 10, 11, 9, 12, 13])
    trig = levels.breakout_triggers(bars, lookback=3, confirm=2)
    assert not trig.iloc[3]                     # single close above
    assert bool(trig.iloc[6])                   # two consecutive closes above the frozen level


def test_stop_floor_pushes_tight_stops_down():
    # swing low only 0.2% below entry -> floored to >= 1% below entry
    entry = 100.0
    near_swing = 99.8
    stop = levels.stop_from_swing(entry, near_swing, min_stop_frac=0.01)
    assert stop <= entry * 0.99 + 1e-9


def test_stop_uses_swing_when_far_enough():
    entry = 100.0
    far_swing = 97.0
    stop = levels.stop_from_swing(entry, far_swing, min_stop_frac=0.01)
    assert stop == 97.0
