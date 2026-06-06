"""Trigger logic and the first-crossings debounce."""

import numpy as np
import pandas as pd

from falling_knife import data, triggers


def test_close_to_close_detects_known_drop():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100, 100, 96, 96.0], index=idx)  # -4% on day 2
    ohlc = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close})
    ret = data.daily_returns(ohlc)
    sig = triggers.close_to_close(ret, threshold=-0.03)
    assert bool(sig.iloc[2]) and not bool(sig.iloc[1])


def test_first_crossings_keeps_only_rising_edge():
    s = pd.Series([False, True, True, True, False, True], dtype=bool)
    fresh = triggers.first_crossings(s, cooldown=0)
    assert list(fresh) == [False, True, False, False, False, True]


def test_first_crossings_cooldown():
    # Two events 2 bars apart; cooldown=5 should suppress the second.
    s = pd.Series([True, False, True, False, False, False, False], dtype=bool)
    fresh = triggers.first_crossings(s, cooldown=5)
    assert list(fresh) == [True, False, False, False, False, False, False]


def test_registry_runs_all(synth_ohlc):
    ret = data.daily_returns(synth_ohlc)
    for name, fn in triggers.TRIGGERS.items():
        sig = fn(ret)
        assert sig.dtype == bool and len(sig) == len(ret)
