"""The DST calendar, synthetic tape, and real fetch are well-formed and deterministic."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daylight_saving import data  # noqa: E402


# ---------------------------------------------------------------------------
# DST calendar tests
# ---------------------------------------------------------------------------
def test_dst_mondays_are_mondays():
    """Every date returned by dst_mondays must be a Monday (dayofweek == 0)."""
    tbl = data.dst_mondays(1980, 2026)
    assert (tbl["date"].dt.dayofweek == 0).all(), \
        "Some DST dates are not Mondays"


def test_dst_mondays_seasons():
    """Each year produces one spring and one fall Monday."""
    tbl = data.dst_mondays(2000, 2023)
    for yr, grp in tbl.groupby("year"):
        seasons = set(grp["season"].tolist())
        assert seasons == {"spring", "fall"}, f"year {yr}: got seasons {seasons}"


def test_dst_rule_change_2007():
    """After 2007 spring switches to March; before it is April."""
    tbl = data.dst_mondays(2005, 2008)
    pre_spring  = tbl[(tbl["year"] == 2006) & (tbl["season"] == "spring")]["date"].iloc[0]
    post_spring = tbl[(tbl["year"] == 2007) & (tbl["season"] == "spring")]["date"].iloc[0]
    assert pre_spring.month == 4, f"2006 spring should be April, got {pre_spring.month}"
    assert post_spring.month == 3, f"2007 spring should be March, got {post_spring.month}"


def test_known_dst_dates():
    """Spot-check known US DST change dates (Monday after the Sunday change)."""
    tbl = data.dst_mondays(2023, 2024)
    # 2023: spring = 2nd Sunday of March = 2023-03-12 → Monday = 2023-03-13
    # 2023: fall   = 1st Sunday of November = 2023-11-05 → Monday = 2023-11-06
    spring_2023 = tbl[(tbl["year"] == 2023) & (tbl["season"] == "spring")]["date"].iloc[0]
    fall_2023   = tbl[(tbl["year"] == 2023) & (tbl["season"] == "fall")]["date"].iloc[0]
    assert spring_2023 == data.pd.Timestamp("2023-03-13"), spring_2023
    assert fall_2023   == data.pd.Timestamp("2023-11-06"), fall_2023


# ---------------------------------------------------------------------------
# Synthetic tape tests
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_tape):
    frame, truth = null_tape
    assert list(frame.columns) == ["ret", "is_dst_monday"]
    assert len(frame) == truth["n_days"]
    assert frame["ret"].notna().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(start_year=1990, end_year=2010, dst_effect=0.0, seed=7)
    b, _ = data.synthetic_daily(start_year=1990, end_year=2010, dst_effect=0.0, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(start_year=1990, end_year=2010, dst_effect=0.0, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_dst_effect_shifts_mean(null_tape, signal_tape):
    """Planting a negative DST effect should lower the mean of DST-Monday returns."""
    frame0, _ = null_tape
    frame1, _ = signal_tape
    dst0 = frame0[frame0["is_dst_monday"]]["ret"].mean()
    dst1 = frame1[frame1["is_dst_monday"]]["ret"].mean()
    assert dst1 < dst0, "Signal tape DST mean should be lower than null tape DST mean"


def test_null_dst_mean_near_other_mondays(null_tape):
    """On the null tape (dst_effect=0) DST Mondays should have a mean close to other days."""
    frame, _ = null_tape
    dst   = frame[frame["is_dst_monday"]]["ret"].mean()
    other = frame[~frame["is_dst_monday"]]["ret"].mean()
    # No planted effect: gap should be small relative to daily vol (~1.1%)
    assert abs(dst - other) < 0.005, \
        f"Null tape DST-other gap too large: {(dst - other)*1e4:.1f} bps"


def test_number_of_dst_mondays_plausible(null_tape):
    """There should be roughly 2 DST Mondays per year (spring + fall)."""
    frame, truth = null_tape
    years = 2025 - 1980 + 1
    n_dst = frame["is_dst_monday"].sum()
    assert 0.9 * 2 * years <= n_dst <= 1.1 * 2 * years, \
        f"Expected ~{2*years} DST Mondays, got {n_dst}"


# ---------------------------------------------------------------------------
# Real fetch cache tests
# ---------------------------------------------------------------------------
def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive(null_tape, signal_tape):
    frame0, _ = null_tape
    frame1, _ = signal_tape
    assert data.fingerprint(frame0) == data.fingerprint(frame0)
    assert data.fingerprint(frame0) != data.fingerprint(frame1)
