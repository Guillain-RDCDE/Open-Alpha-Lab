"""Tests for the data layer — synthetic tape properties and cache behaviour."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mark_twain import data  # noqa: E402


def test_synthetic_shape(null_tape):
    ret, truth = null_tape
    # Roughly n_years * 252 trading days.
    assert len(ret) >= truth["n_years"] * 248
    assert len(ret) <= truth["n_years"] * 256
    assert (ret > -1.0).all(), "no return should wipe out a position"
    assert ret.index.is_monotonic_increasing


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=20, october_penalty_bp=0.0, seed=7)
    b, _ = data.synthetic_daily(n_years=20, october_penalty_bp=0.0, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_daily(n_years=20, october_penalty_bp=0.0, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_null_has_no_october_effect(null_tape):
    """With october_penalty=0, October and, say, July should have similar daily means."""
    ret, _ = null_tape
    oct_mean = ret[ret.index.month == 10].mean()
    jul_mean = ret[ret.index.month == 7].mean()
    # No penalty → the difference should be small (< 50 bps in daily terms).
    assert abs(oct_mean - jul_mean) < 0.005


def test_synthetic_penalised_october_is_lower(penalised_tape, null_tape):
    """With a planted penalty, October should be materially worse than in the null."""
    ret_pen, _ = penalised_tape
    ret_null, _ = null_tape
    oct_pen = ret_pen[ret_pen.index.month == 10].mean()
    oct_null = ret_null[ret_null.index.month == 10].mean()
    assert oct_pen < oct_null - 0.005, "penalty tape must have lower October mean"


def test_fetch_daily_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_sensitive(null_tape):
    ret, _ = null_tape
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _ = data.synthetic_daily(n_years=60, seed=99)
    assert data.fingerprint(ret) != data.fingerprint(other)
