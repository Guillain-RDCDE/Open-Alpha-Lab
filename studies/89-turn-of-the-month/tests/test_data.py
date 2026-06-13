"""Data-layer invariants for Study 89 (Turn-of-the-Month)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from turn_of_the_month import data  # noqa: E402


def test_shape_and_columns(null_walk):
    frame, truth = null_walk
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(n_days=1000, bump=0.001, seed=7)
    b, _ = data.synthetic_daily(n_days=1000, bump=0.001, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_tom_fraction_is_a_minority(null_walk):
    _, truth = null_walk
    # The [-1,+3] window is ~4 of ~22 business days per month -> a clear minority.
    assert 0.10 < truth["frac_tom"] < 0.30


def test_bump_only_lifts_close(null_walk, planted_tom):
    # The planted bump raises the terminal level vs the same-seed null walk.
    null_frame, _ = null_walk
    planted_frame, _ = planted_tom
    assert planted_frame["close"].iloc[-1] > null_frame["close"].iloc[-1]


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, bump=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, bump=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
