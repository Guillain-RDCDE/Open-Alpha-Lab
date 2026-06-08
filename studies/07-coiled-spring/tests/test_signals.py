"""The rule must be causal, obey the book's hard gates, and find planted structure."""

import numpy as np
import pandas as pd
import pytest

from coiled_spring import data, signals
from coiled_spring.signals import Params, find_signals


def test_ema_matches_pandas_ewm():
    s = pd.Series(np.linspace(10, 20, 50) + np.sin(np.arange(50)))
    got = signals.ema(s, 20)
    want = s.ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_pivot_is_strict_local_max():
    high = np.array([1, 2, 3, 5, 3, 2, 1, 2, 4, 4, 2, 1], dtype=float)
    mask = signals._confirmed_pivot_highs(high, halfwin=3)
    assert mask[3]              # the clean peak at index 3
    assert not mask[8] and not mask[9]   # a flat double-top is not a strict max


def test_no_lookahead_entry_is_after_breakout():
    frames, _ = data.synthetic_universe(seed=1)
    f = next(iter(frames.values()))
    sig = find_signals(f)
    pos = {d: i for i, d in enumerate(f.index)}
    for _, s in sig.iterrows():
        if pd.isna(s["entry_date"]):
            continue
        # entry is strictly after the breakout, which is strictly after the pivot.
        assert pos[s["entry_date"]] == pos[s["breakout_date"]] + 1
        assert pos[s["breakout_date"]] > pos[s["pivot_date"]]


def test_volume_gate_blocks_thin_breakouts():
    frames, _ = data.synthetic_universe(seed=2)
    f = next(iter(frames.values()))
    loose = find_signals(f, Params(vol_mult=0.0))
    strict = find_signals(f, Params(vol_mult=3.0))
    assert len(strict) <= len(loose)            # a higher volume bar can only remove signals
    if len(strict):
        assert (strict["vol_ratio"] >= 3.0).all()


def test_breakout_above_pivot_and_holds_ema():
    frames, _ = data.synthetic_universe(seed=0)
    for tk, f in frames.items():
        sig = find_signals(f)
        e = signals.ema(f["Close"], 20)
        for _, s in sig.iterrows():
            assert s["pivot_level"] > 0
            # the breakout close clears the pivot level
            assert f.loc[s["breakout_date"], "Close"] > s["pivot_level"]
            # and the breakout bar itself was not below the EMA (the pullback held it)
            assert f.loc[s["breakout_date"], "Close"] >= e.loc[s["breakout_date"]] - 1e-9


def test_detector_recovers_planted_springboards(synthetic):
    from coiled_spring import robustness
    frames, truth = synthetic
    rec = robustness.detector_recall(frames, truth)
    # The detector must find a clear majority-ish of planted breakouts (machinery sanity).
    assert rec["recall"] > 0.4, rec


def test_empty_on_too_short_frame():
    idx = pd.bdate_range("2020-01-01", periods=10)
    f = pd.DataFrame({c: np.linspace(1, 2, 10) for c in ["Open", "High", "Low", "Close"]}, index=idx)
    f["Volume"] = 1e6
    assert find_signals(f).empty
