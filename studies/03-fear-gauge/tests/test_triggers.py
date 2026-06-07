"""Trigger logic: the level, the spike, the base split, and the debounce."""

import pandas as pd

from fear_gauge import triggers


def _vix(levels):
    s = pd.Series(levels, index=pd.bdate_range("2020-01-01", periods=len(levels)), dtype=float)
    df = pd.DataFrame({"Close": s})
    df["vix_prev"] = df["Close"].shift(1)
    df["vix_chg"] = df["Close"] / df["vix_prev"] - 1.0
    return df


def test_level_threshold():
    vix = _vix([20, 28, 31, 55, 25])
    sig = triggers.level(vix, 30)
    assert list(sig) == [False, False, True, True, False]


def test_spike_detects_30pct_jump():
    vix = _vix([20, 20, 27, 27])  # +35% on day 2
    sig = triggers.spike(vix, 0.30)
    assert bool(sig.iloc[2]) and not bool(sig.iloc[1])


def test_spike_from_base_splits_regimes():
    # Same +40% jump, once from a low base (14->19.6), once from a high (40->56).
    vix = _vix([14, 19.6, 40, 56])
    low = triggers.spike_from_base(vix, base_max=20)
    high = triggers.spike_from_base(vix, base_min=30)
    assert bool(low.iloc[1]) and not bool(low.iloc[3])
    assert bool(high.iloc[3]) and not bool(high.iloc[1])


def test_first_crossings_keeps_only_rising_edge():
    s = pd.Series([False, True, True, True, False, True], dtype=bool)
    fresh = triggers.first_crossings(s, cooldown=0)
    assert list(fresh) == [False, True, False, False, False, True]


def test_first_crossings_cooldown():
    s = pd.Series([True, False, True, False, False, False, False], dtype=bool)
    fresh = triggers.first_crossings(s, cooldown=5)
    assert list(fresh) == [True, False, False, False, False, False, False]


def test_registry_runs_all(synth_gauge):
    for name, fn in triggers.TRIGGERS.items():
        sig = fn(synth_gauge)
        assert sig.dtype == bool and len(sig) == len(synth_gauge)
