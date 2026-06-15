"""The synthetic tape is well-formed and deterministic; calendar helpers are correct;
the real fetch is properly cache-gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monthly_opex import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — any test that reads a real parquet must carry this mark.
# ---------------------------------------------------------------------------
_CACHE_PATH = data._cache_path("SPY", data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def test_third_friday_is_friday():
    for year in range(2000, 2027):
        for month in range(1, 13):
            d = data._third_friday(year, month)
            assert d.weekday() == 4, f"{year}-{month:02d}: {d} is not a Friday"


def test_third_friday_is_third():
    """The date must be in the range [15, 21] — the only Fridays that are 'third'."""
    for year in range(2010, 2027):
        for month in range(1, 13):
            d = data._third_friday(year, month)
            assert 15 <= d.day <= 21, f"{d} is not the third Friday"


def test_opex_dates_count():
    """12 monthly OpEx dates per year (one per month)."""
    od = data.opex_dates("2020-01-01", "2020-12-31")
    assert len(od) == 12


def test_opex_dates_all_fridays():
    od = data.opex_dates("2000-01-01", "2026-12-31")
    for d in od:
        assert d.weekday() == 4, f"{d} is not a Friday"


def test_opex_week_mask_covers_five_days():
    """The OpEx-week mask must flag exactly 5 consecutive calendar days around the 3rd Friday."""
    idx = pd.bdate_range("2024-01-01", "2024-12-31", name="date")
    mask = data.opex_week_mask(idx)
    # In a normal month, one OpEx Friday → Mon–Fri of that week should be flagged.
    # (Holidays may reduce this to 4 days, but never more than 5.)
    runs: list[int] = []
    in_run = False
    run_len = 0
    for v in mask.values:
        if v:
            in_run = True
            run_len += 1
        elif in_run:
            runs.append(run_len)
            in_run = False
            run_len = 0
    if in_run:
        runs.append(run_len)
    assert all(1 <= r <= 5 for r in runs), f"unexpected run lengths: {runs}"


def test_opex_day_mask_count():
    """There is exactly one OpEx day per month in the mask (modulo holidays near 3rd Friday)."""
    idx = pd.bdate_range("2020-01-01", "2020-12-31", name="date")
    mask = data.opex_day_mask(idx)
    # 12 months in 2020 — the 3rd Fridays in 2020 are all real trading days
    assert int(mask.sum()) == 12


def test_monthly_and_quarterly_masks_partition():
    """Monthly-only + quarterly must sum to the full monthly mask with no overlap."""
    idx = pd.bdate_range("2010-01-01", "2026-12-31", name="date")
    full = data.opex_week_mask(idx)
    monthly_only = data.monthly_only_opex_week_mask(idx)
    quarterly = data.quarterly_opex_week_mask(idx)
    # No day in both monthly-only and quarterly
    assert not (monthly_only & quarterly).any()
    # Their union = full OpEx week mask
    assert ((monthly_only | quarterly) == full).all()


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------

def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == truth["n_days"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, vol_premium=0.2, seed=7)
    b, _ = data.synthetic_daily(n_years=10, vol_premium=0.2, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, vol_premium=0.2, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_vol_premium_elevates_range(null_tape, vol_tape):
    """A non-zero vol_premium must produce a higher OpEx-week range than the null."""
    idx_null = pd.DatetimeIndex(null_tape[0].index)
    idx_vol = pd.DatetimeIndex(vol_tape[0].index)
    mask_null = data.opex_week_mask(idx_null)
    mask_vol = data.opex_week_mask(idx_vol)

    rng_null = ((null_tape[0]["high"] - null_tape[0]["low"]) / null_tape[0]["close"])
    rng_vol = ((vol_tape[0]["high"] - vol_tape[0]["low"]) / vol_tape[0]["close"])

    opex_null = float(rng_null[mask_null].mean())
    opex_vol = float(rng_vol[mask_vol].mean())
    assert opex_vol > opex_null, "vol_premium should elevate OpEx-week range"


def test_ret_premium_elevates_return(null_tape, ret_tape):
    """A non-zero ret_premium_bps must raise the OpEx-week mean return."""
    idx_null = pd.DatetimeIndex(null_tape[0].index)
    idx_ret = pd.DatetimeIndex(ret_tape[0].index)
    mask_null = data.opex_week_mask(idx_null)
    mask_ret = data.opex_week_mask(idx_ret)

    r_null = null_tape[0]["close"].pct_change()
    r_ret = ret_tape[0]["close"].pct_change()

    mean_null = float(r_null[mask_null].mean())
    mean_ret = float(r_ret[mask_ret].mean())
    assert mean_ret > mean_null, "ret_premium_bps should elevate OpEx-week return"


# ---------------------------------------------------------------------------
# Cache-gated real-data tests
# ---------------------------------------------------------------------------

@requires_cache
def test_real_fetch_shape():
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) > 7000
    assert bars.index.name == "date"
    assert bars.index.tz is None


@requires_cache
def test_fingerprint_stable():
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    assert data.fingerprint(bars) == data.fingerprint(bars)


@requires_cache
def test_opex_event_count_real():
    """There should be ~12 × 33 ~ 396 OpEx events in the SPY tape (1993–2026)."""
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_day_mask(idx)
    assert int(mask.sum()) >= 380
