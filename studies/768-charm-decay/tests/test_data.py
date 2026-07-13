"""Data-layer invariants — pure calendar arithmetic, no network, deterministic."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from charm_decay import data  # noqa: E402


def test_third_friday_known_dates():
    # 2024-06-21 is the 3rd Friday of June 2024 (quarterly triple-witching)
    assert data._third_friday(2024, 6) == pd.Timestamp("2024-06-21")
    # 2020-01-17 is the 3rd Friday of January 2020
    assert data._third_friday(2020, 1) == pd.Timestamp("2020-01-17")
    assert data._third_friday(2024, 6).weekday() == 4  # Friday


def test_opex_dates_twelve_per_year():
    od = data.opex_dates("2015-01-01", "2015-12-31")
    assert len(od) == 12
    assert (pd.DatetimeIndex(od).weekday == 4).all()


def test_is_quarterly_opex():
    assert data.is_quarterly_opex(pd.Timestamp("2024-06-21"))
    assert not data.is_quarterly_opex(pd.Timestamp("2024-05-17"))


def test_pre_post_windows_disjoint_and_sized():
    bars, _ = data.synthetic_daily(n_years=10, seed=768)
    idx = pd.DatetimeIndex(bars.index)
    pre = data.pre_opex_mask(idx, ndays=5)
    post = data.post_opex_mask(idx, ndays=5)
    # Pre and post windows never overlap
    assert not (pre.values & post.values).any()
    # Roughly 5 days per OpEx event present in the tape
    n_events = len(data.opex_dates(str(idx.min().date()), str(idx.max().date())))
    assert 4.0 <= pre.sum() / n_events <= 5.0


def test_shift_moves_the_window():
    bars, _ = data.synthetic_daily(n_years=10, seed=768)
    idx = pd.DatetimeIndex(bars.index)
    base = data.pre_opex_mask(idx, ndays=5)
    shifted = data.charm_window_mask(idx, lo=-4, hi=0, shift=10)
    # A 10-day shift must change which days are flagged
    assert (base.values != shifted.values).any()
    assert base.sum() > 0 and shifted.sum() > 0


def test_synthetic_null_vs_planted_differs():
    null_bars, _ = data.synthetic_daily(n_years=10, pre_drift_bps=0.0, seed=768)
    drift_bars, _ = data.synthetic_daily(n_years=10, pre_drift_bps=10.0, seed=768)
    # Same noise seed, different planted drift → different close paths
    assert not null_bars["close"].equals(drift_bars["close"])


def test_fingerprint_deterministic():
    bars, _ = data.synthetic_daily(n_years=5, seed=768)
    assert data.fingerprint(bars) == data.fingerprint(bars)
    assert len(data.fingerprint(bars)) == 12
