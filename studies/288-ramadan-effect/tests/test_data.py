"""Tests for the data layer of Study 288 (Ramadan-Effect).

All tests are offline and deterministic -- they use the hardcoded Ramadan window
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ramadan_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Ramadan window table
# ---------------------------------------------------------------------------
def test_table_columns(ramadan_table):
    expected = {"hijri_year", "start", "end"}
    assert expected.issubset(set(ramadan_table.columns))


def test_table_covers_range(ramadan_table):
    """Windows span 2000 through 2026 (Gregorian)."""
    assert ramadan_table["start"].min().year <= 2000
    assert ramadan_table["end"].max().year >= 2026


def test_table_dates_are_timestamps(ramadan_table):
    assert pd.api.types.is_datetime64_any_dtype(ramadan_table["start"])
    assert pd.api.types.is_datetime64_any_dtype(ramadan_table["end"])


def test_windows_are_about_a_month_long(ramadan_table):
    """Each Ramadan window is 28-31 days (a lunar month)."""
    lengths = (ramadan_table["end"] - ramadan_table["start"]).dt.days + 1
    assert lengths.between(28, 31).all()


def test_windows_drift_earlier(ramadan_table):
    """Ramadan drifts ~11 days earlier each Gregorian year (monotone-ish)."""
    starts = ramadan_table.sort_values("hijri_year")["start"].reset_index(drop=True)
    # day-of-year should generally decrease year over year (with wraparound)
    doy = starts.dt.dayofyear.to_numpy()
    # at least 80% of consecutive steps move earlier or wrap (jump up by ~350)
    diffs = np.diff(doy)
    earlier_or_wrap = ((diffs < 0) | (diffs > 300)).mean()
    assert earlier_or_wrap >= 0.8


def test_windows_do_not_overlap(ramadan_table):
    """Consecutive Ramadan windows must not overlap."""
    t = ramadan_table.sort_values("start").reset_index(drop=True)
    for i in range(len(t) - 1):
        assert t.loc[i, "end"] < t.loc[i + 1, "start"]


def test_ramadan_table_is_copy():
    a = data.ramadan_windows()
    b = data.ramadan_windows()
    a.loc[0, "hijri_year"] = -1
    assert b.loc[0, "hijri_year"] != -1


# ---------------------------------------------------------------------------
# Overlap fraction & month labelling
# ---------------------------------------------------------------------------
def test_overlap_fraction_in_range():
    months = pd.date_range("2015-01-31", periods=48, freq="ME")
    for m in months:
        f = data.ramadan_overlap_fraction(m)
        assert 0.0 <= f <= 1.0


def test_known_ramadan_month_is_flagged():
    """April 2022 is mostly Ramadan (2 Apr - 1 May 2022) -> should be flagged."""
    lab = data.label_months([pd.Timestamp("2022-04-30")], threshold=0.5)
    assert bool(lab["ramadan"].iloc[0]) is True


def test_clearly_non_ramadan_month_is_not_flagged():
    """December 2022 has no Ramadan overlap."""
    lab = data.label_months([pd.Timestamp("2022-12-31")], threshold=0.5)
    assert bool(lab["ramadan"].iloc[0]) is False


def test_label_months_count_reasonable():
    """Over ~10 years there should be roughly one Ramadan month per year."""
    months = pd.date_range("2015-01-31", periods=120, freq="ME")
    lab = data.label_months(months, threshold=0.5)
    n_ram = int(lab["ramadan"].sum())
    assert 8 <= n_ram <= 12


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_months"]
    assert set(df.columns) >= {"ret", "overlap", "ramadan"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, Ramadan/rest mean returns should be close (no planted effect)."""
    df, _ = data.synthetic_monthly(n_months=3000, premium_bps=0.0, seed=288)
    ram = df.loc[df["ramadan"], "ret"].mean()
    rest = df.loc[~df["ramadan"], "ret"].mean()
    assert abs(ram - rest) < 0.01


def test_synthetic_signal_creates_difference(synthetic_signal):
    """A planted premium should lift Ramadan-month returns above the rest."""
    df, _ = synthetic_signal
    ram = df.loc[df["ramadan"], "ret"].mean()
    rest = df.loc[~df["ramadan"], "ret"].mean()
    assert ram > rest + 0.005


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=10)
    df2, _ = data.synthetic_monthly(n_months=100, premium_bps=0.0, seed=11)
    df1 = df1.rename(columns={"ret": "mena_ret"})
    df2 = df2.rename(columns={"ret": "mena_ret"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real-tape loader is offline-safe
# ---------------------------------------------------------------------------
def test_fetch_proxy_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_proxy raises FileNotFoundError."""
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_proxy(fetch=False, cache_path=missing)
