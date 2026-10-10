"""Data-layer tests for Study 1019 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volclock import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes not installed — synthetic tests cover the logic")


def test_synthetic_tape_is_deterministic():
    a, _ = data.synthetic_tape(n_days=800, seed=1019)
    b, _ = data.synthetic_tape(n_days=800, seed=1019)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_tape_is_seed_sensitive():
    a, _ = data.synthetic_tape(n_days=800, seed=1019)
    b, _ = data.synthetic_tape(n_days=800, seed=1020)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_tape_shape_index_and_ohlc_order():
    tape, truth = data.synthetic_tape(n_days=1500, seed=1019)
    assert list(tape.columns) == ["open", "high", "low", "close", "volume"]
    assert len(tape) == truth["n_days"] == 1500
    assert isinstance(tape.index, pd.DatetimeIndex) and tape.index.is_monotonic_increasing
    assert tape.index[-1] < pd.Timestamp("2262-01-01")     # inside pandas' ns horizon
    assert (tape["high"] >= tape[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (tape["low"] <= tape[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (tape > 0).all().all()


def test_signal_strength_is_recorded_and_clipped():
    _, t1 = data.synthetic_tape(n_days=300, signal_strength=1.0)
    _, t0 = data.synthetic_tape(n_days=300, signal_strength=0.0)
    _, tc = data.synthetic_tape(n_days=300, signal_strength=3.0)
    assert t1["signal_strength"] == 1.0 and t0["signal_strength"] == 0.0
    assert tc["signal_strength"] == 1.0


def test_raw_tails_do_not_depend_on_the_knob():
    """The null is *matched*: only the link to volume moves, not the fat tails themselves."""
    _, t1 = data.synthetic_tape(n_days=300, signal_strength=1.0)
    _, t0 = data.synthetic_tape(n_days=300, signal_strength=0.0)
    assert t1["theoretical_excess_kurtosis"] == t0["theoretical_excess_kurtosis"] > 1.0


def test_synthetic_volume_trends_up_so_detrending_matters():
    tape, _ = data.synthetic_tape(n_days=5000, seed=1019)
    lv = np.log(tape["volume"])
    assert lv.iloc[-500:].mean() - lv.iloc[:500].mean() > 1.0


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_tape(n_days=400, seed=1019)
    b, _ = data.synthetic_tape(n_days=400, seed=1020)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_unknown_tape_is_refused():
    with pytest.raises(KeyError):
        data.load_tape("ftse")


def test_as_of_is_a_complete_year_and_not_in_the_future():
    asof = pd.Timestamp(data.AS_OF)
    assert asof.month == 12 and asof.day == 31
    assert asof < pd.Timestamp.today()


@needs_real
def test_real_tapes_load_and_are_pinned():
    for name in data.TAPES:
        t = data.load_tape(name)
        assert list(t.columns) == ["open", "high", "low", "close", "volume"]
        assert t.index[0] >= pd.Timestamp(data.START)
        assert t.index[-1] == pd.Timestamp(data.AS_OF)
        assert t.index.is_monotonic_increasing and not t.index.has_duplicates
        assert (t["high"] >= t["low"]).all() and t["close"].notna().all()
        assert len(data.sha_pin(name)) == 64


@needs_real
def test_zero_volume_prints_become_missing_not_zero():
    t = data.load_tape("nasdaq")
    assert (t["volume"] > 0).sum() + t["volume"].isna().sum() == len(t)
    assert t["volume"].isna().sum() == 2


@needs_real
def test_daily_rf_is_small_positive_and_covers_december_2018():
    t = data.load_tape("sp500")
    rf = data.load_daily_rf(t.index)
    assert rf.notna().all()
    assert (rf >= 0).all() and rf.max() < 0.001
    # December 2018 is past the end of the French file: November's monthly rate carries over
    assert rf.loc["2018-12"].sum() == pytest.approx(rf.loc["2018-11"].sum(), rel=1e-9)
