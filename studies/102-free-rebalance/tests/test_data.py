"""Data-layer invariants for Study 102 (Free-Rebalance)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from free_rebalance import data  # noqa: E402


def test_shape_and_columns(mean_revert):
    frame, truth = mean_revert
    assert list(frame.columns) == ["A", "B"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_meanrevert(n_days=1000, seed=7)
    b, _ = data.synthetic_meanrevert(n_days=1000, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_expected_signs_recorded(mean_revert, trending):
    assert mean_revert[1]["expected_bonus_sign"] == +1
    assert trending[1]["expected_bonus_sign"] == -1


def test_trend_lead_outgrows_lag(trending):
    frame, _ = trending
    # The leading asset must finish well above the laggard (it trends away).
    assert frame["A"].iloc[-1] > frame["B"].iloc[-1]


def test_meanrevert_legs_stay_close(mean_revert):
    frame, _ = mean_revert
    # Equal drift + mean reversion: the two legs don't run away from each other.
    log_ratio = np.log(frame["A"] / frame["B"])
    assert abs(log_ratio.iloc[-1]) < 1.0  # within e^1 of each other


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_meanrevert(n_days=500, seed=1)
    b, _ = data.synthetic_meanrevert(n_days=500, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)
