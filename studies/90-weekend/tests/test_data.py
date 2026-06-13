"""Data-layer invariants for Study 90 (Weekend / day-of-week effect)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weekend import data  # noqa: E402


def test_shape_and_columns(null_tape):
    frame, truth = null_tape
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame["close"] > 0).all()


def test_index_is_business_days(null_tape):
    frame, _ = null_tape
    # bdate_range is Mon-Fri only — no Saturday(5)/Sunday(6) bars.
    assert set(frame.index.dayofweek.unique()) <= {0, 1, 2, 3, 4}


def test_determinism():
    a, _ = data.synthetic_daily(n_days=1000, dow_effect=0.0010, seed=7)
    b, _ = data.synthetic_daily(n_days=1000, dow_effect=0.0010, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_planted_bump(null_tape):
    _, truth = null_tape
    assert all(v == 0.0 for v in truth["planted_bump"].values())


def test_planted_bump_is_present(planted_tape):
    _, truth = planted_tape
    bump = truth["planted_bump"]
    # Monday pushed down, Tuesday up, others untouched.
    assert bump["Mon"] < 0
    assert bump["Tue"] > 0
    assert bump["Wed"] == bump["Thu"] == bump["Fri"] == 0.0


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, dow_effect=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, dow_effect=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
