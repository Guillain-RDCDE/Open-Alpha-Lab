"""The retrograde table, the synthetic tape, and the real fetch are well-formed."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mercury_retrograde import data  # noqa: E402


# ---------------------------------------------------------------------------
# Retrograde table tests
# ---------------------------------------------------------------------------
def test_retrograde_dates_are_sorted():
    """The retrograde date range must be in chronological order."""
    days = data.retrograde_dates()
    assert days.is_monotonic_increasing


def test_retrograde_dates_no_gaps_within_period():
    """Within each retrograde period the daily sequence must be consecutive."""
    for s, e in data.RETROGRADE_PERIODS:
        ts, te = pd.Timestamp(s), pd.Timestamp(e)
        expected = pd.date_range(ts, te, freq="D")
        found = data.retrograde_dates(start=s, end=e)
        assert len(found) == len(expected), f"Gap in period {s}→{e}"


def test_retrograde_fraction_is_realistic():
    """Mercury is retrograde ~19% of the time — must be between 15% and 25%."""
    days_all = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    retro = data.retrograde_dates("2000-01-01", "2025-12-31")
    frac = len(retro) / len(days_all)
    assert 0.15 <= frac <= 0.25, f"Fraction {frac:.3f} outside realistic range"


def test_retrograde_table_no_overlaps():
    """No two retrograde periods should overlap."""
    periods = sorted(data.RETROGRADE_PERIODS, key=lambda x: x[0])
    for i in range(len(periods) - 1):
        end_i = pd.Timestamp(periods[i][1])
        start_next = pd.Timestamp(periods[i + 1][0])
        assert end_i < start_next, (
            f"Overlap between {periods[i]} and {periods[i + 1]}"
        )


def test_is_retrograde_alignment(null_tape):
    """is_retrograde mask must be aligned to the return series index."""
    ret, mask, _ = null_tape
    assert len(mask) == len(ret)
    assert (mask.index == ret.index).all()


def test_is_retrograde_fraction(null_tape):
    """Retrograde fraction on synthetic tape should be ~15-25%."""
    _, mask, truth = null_tape
    frac = float(mask.mean())
    assert 0.15 <= frac <= 0.25, f"Fraction {frac:.3f}"


# ---------------------------------------------------------------------------
# Synthetic tape tests
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_determinism():
    """Shape is correct and two calls with the same seed produce the same series."""
    a, mask_a, truth = data.synthetic_daily(n_years=10, seed=42)
    b, mask_b, _ = data.synthetic_daily(n_years=10, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert len(a) == truth["n_obs"]


def test_synthetic_different_seeds_differ():
    a, _, _ = data.synthetic_daily(n_years=10, seed=1)
    b, _, _ = data.synthetic_daily(n_years=10, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_penalty_shifts_retrograde_mean(null_tape, planted_tape):
    """With a planted -20 bps/day penalty the retrograde mean must be lower."""
    ret_null, mask_null, _ = null_tape
    ret_plant, mask_plant, _ = planted_tape
    mean_null = float(ret_null[mask_null].mean() * 1e4)
    mean_plant = float(ret_plant[mask_plant].mean() * 1e4)
    # The planted tape should have a retrograde mean substantially lower
    assert mean_plant < mean_null - 10, (
        f"planted mean {mean_plant:.2f} bps vs null {mean_null:.2f} bps"
    )


def test_synthetic_truth_records_params():
    _, _, truth = data.synthetic_daily(n_years=5, retro_penalty_bp=-15.0, seed=7)
    assert truth["retro_penalty_bp"] == -15.0
    assert truth["n_years"] == 5
    assert truth["seed"] == 7
    assert truth["n_retro_days"] > 0


# ---------------------------------------------------------------------------
# Real fetch — cache-only / error path
# ---------------------------------------------------------------------------
def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive(null_tape):
    ret, _, _ = null_tape
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _, _ = data.synthetic_daily(n_years=25, seed=999)
    assert data.fingerprint(ret) != data.fingerprint(other)
