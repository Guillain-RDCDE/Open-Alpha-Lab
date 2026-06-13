"""Data-layer invariants for Study 95 (Holiday-Cheer), incl. the holiday-derivation test."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from holiday_cheer import data  # noqa: E402


def test_shape_and_columns(planted_tape):
    frame, truth = planted_tape
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"]
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(planted=True, seed=7)
    b, _ = data.synthetic_daily(planted=True, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(planted=True, seed=1)
    b, _ = data.synthetic_daily(planted=True, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_derivation_finds_no_weekend_holidays(planted_tape):
    """Weekends are never flagged as holidays (bdate_range already excludes them)."""
    frame, _ = planted_tape
    hols = data.derive_market_holidays(frame.index)
    assert all(h.weekday() < 5 for h in hols)  # Mon-Fri only


def test_derivation_finds_known_holidays():
    """The gap-derivation recovers the engineered July-4 and Dec-25 weekday closures.

    The synthetic tape *removes* the weekday occurrences of a fixed holiday list; the
    derivation must rediscover them purely from the gaps in the trading index - no external
    calendar consulted. Over a 30-year span we must find many July-4 and Dec-25 holidays.
    """
    frame, _ = data.synthetic_daily(planted=False, seed=95, n_years=30)
    hols = data.derive_market_holidays(frame.index)
    jul4 = [h for h in hols if h.month == 7 and h.day == 4]
    dec25 = [h for h in hols if h.month == 12 and h.day == 25]
    jan1 = [h for h in hols if h.month == 1 and h.day == 1]
    # Each falls on a weekday in well over half of 30 years.
    assert len(jul4) >= 15
    assert len(dec25) >= 15
    assert len(jan1) >= 15


def test_pre_holiday_is_a_trading_day_before_a_holiday():
    """Every derived pre-holiday day is a real trading day, immediately before a holiday."""
    frame, _ = data.synthetic_daily(planted=False, seed=95)
    idx = frame.index
    pre = data.derive_pre_holiday(idx)
    hols = set(data.derive_market_holidays(idx).normalize())
    trading = set(idx.normalize())
    for d in pre:
        assert d in trading  # it is a trading day
        # the next calendar weekday after it that is missing is a holiday
        nxt = d + pd.tseries.offsets.BDay(1)
        # at least one of the following business days up to a week out is a holiday
        assert any((d + pd.tseries.offsets.BDay(k)).normalize() in hols for k in range(1, 6))


def test_pre_holiday_mask_aligns(planted_tape):
    frame, _ = planted_tape
    mask = data.pre_holiday_mask(frame.index)
    assert len(mask) == len(frame)
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_planted_truth_records_bump():
    _, truth = data.synthetic_daily(planted=True, bump=0.005, seed=95)
    assert truth["bump"] == 0.005
    _, truth0 = data.synthetic_daily(planted=False, seed=95)
    assert truth0["bump"] == 0.0
