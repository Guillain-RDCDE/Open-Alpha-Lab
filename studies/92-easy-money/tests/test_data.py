"""Data-layer invariants for Study 92 (Easy-Money)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from easy_money import data  # noqa: E402


def test_shape_and_columns(null_walk):
    frame, truth = null_walk
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 4000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_vol_etp(n_days=1000, carry=0.003, seed=7)
    b, _ = data.synthetic_vol_etp(n_days=1000, carry=0.003, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_jumps(null_walk):
    _, truth = null_walk
    assert truth["n_jumps"] == 0


def test_planted_has_some_jumps(planted_carry):
    _, truth = planted_carry
    # Rare upward spikes — a handful over the sample, not zero, not constant.
    assert 1 <= truth["n_jumps"] < 200


def test_planted_tape_decays(planted_carry):
    # The planted contango bleed should leave the ETP far below where it started.
    frame, _ = planted_carry
    assert frame["close"].iloc[-1] < frame["close"].iloc[0]


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_vol_etp(n_days=500, carry=0.0, seed=1)
    b, _ = data.synthetic_vol_etp(n_days=500, carry=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
