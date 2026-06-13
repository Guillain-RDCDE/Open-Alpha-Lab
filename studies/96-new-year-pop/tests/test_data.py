"""Data-layer invariants for Study 96 (New-Year-Pop)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_year_pop import data  # noqa: E402


def test_shape_and_columns(null_tape):
    frame, truth = null_tape
    assert list(frame.columns) == ["small", "large"]
    assert len(frame) == truth["n_months"] == 36 * 12


def test_index_is_month_end(null_tape):
    frame, _ = null_tape
    # Every bar is a month-end label.
    assert (frame.index.is_month_end).all()


def test_determinism():
    a, _ = data.synthetic_monthly(n_years=10, jan_bump=0.03, seed=7)
    b, _ = data.synthetic_monthly(n_years=10, jan_bump=0.03, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_monthly(n_years=10, jan_bump=0.0, seed=1)
    b, _ = data.synthetic_monthly(n_years=10, jan_bump=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_planted_bump_lifts_january_small_leg(planted_tape):
    frame, truth = planted_tape
    jan = frame.index.month == 1
    # The small leg's January mean must exceed its other-month mean by ~the planted bump.
    lift = frame["small"][jan].mean() - frame["small"][~jan].mean()
    assert lift > truth["jan_bump"] * 0.5


def test_null_has_no_january_lift(null_tape):
    frame, _ = null_tape
    jan = frame.index.month == 1
    lift = frame["small"][jan].mean() - frame["small"][~jan].mean()
    assert abs(lift) < 0.02  # small noise only, no planted seasonal
