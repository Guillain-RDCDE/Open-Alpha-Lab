"""Data-layer invariants for Study 101 (Slow-and-Steady)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slow_and_steady import data  # noqa: E402


def test_shape_and_columns(up_tape):
    frame, truth = up_tape
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(n_days=1000, drift=0.10, seed=7)
    b, _ = data.synthetic_daily(n_days=1000, drift=0.10, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_expected_winner_tracks_drift_sign(up_tape, down_tape):
    assert up_tape[1]["expected_winner"] == "lump_sum"
    assert down_tape[1]["expected_winner"] == "dca"


def test_drift_sign_shows_in_path(up_tape, down_tape):
    # A positive-drift tape ends higher than it starts; negative ends lower (in expectation,
    # and comfortably so over 6000 days at +/-15%/yr).
    assert up_tape[0]["close"].iloc[-1] > up_tape[0]["close"].iloc[0]
    assert down_tape[0]["close"].iloc[-1] < down_tape[0]["close"].iloc[0]


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, drift=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, drift=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
