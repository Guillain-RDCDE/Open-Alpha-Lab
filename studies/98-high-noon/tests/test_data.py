"""Data-layer invariants for Study 98 (High-Noon)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from high_noon import data  # noqa: E402


def test_shape_and_columns(null_walk):
    frame, truth = null_walk
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 8000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(n_days=1000, ath_reversal=1.0, seed=7)
    b, _ = data.synthetic_daily(n_days=1000, ath_reversal=1.0, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_both_tapes_spend_some_time_near_highs(null_walk, planted_reversal):
    # Whatever the knob, a drifting tape spends a non-trivial minority near its highs.
    for _, truth in (null_walk, planted_reversal):
        assert 0.02 < truth["frac_near_high"] < 0.60


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, ath_reversal=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, ath_reversal=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
