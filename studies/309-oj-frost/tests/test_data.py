"""The synthetic tape is well-formed, deterministic, and plants what it claims; the
freeze table is sane; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oj_frost import data  # noqa: E402


def test_synthetic_shape(null_tape):
    frame, truth = null_tape
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"]
    assert (frame["close"] > 0).all()
    assert frame.index.tz is None


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_oj(n_days=2000, freeze_jump=0.1, seed=5)
    b, _ = data.synthetic_oj(n_days=2000, freeze_jump=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_oj(n_days=2000, freeze_jump=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_freeze_table_is_sane():
    fz = data.freeze_dates()
    assert len(fz) >= 8
    assert fz.is_monotonic_increasing
    assert fz.tz is None
    # All freezes fall in the cold months (Nov–Feb).
    assert set(fz.month.unique()) <= {11, 12, 1, 2}


def test_synthetic_freeze_dates_recorded(freeze_tape):
    _, truth = freeze_tape
    assert len(truth["syn_freezes"]) >= 8
    assert truth["freeze_jump"] == 0.20


def test_planted_jump_raises_post_freeze_returns():
    """With a planted jump, the mean return in the window AFTER synthetic freezes exceeds
    the unconditional mean; with no jump it does not (the null)."""
    jump, tj = data.synthetic_oj(freeze_jump=0.20, seed=309)
    flat, tf = data.synthetic_oj(freeze_jump=0.0, seed=309)

    def post_freeze_mean(frame, freezes, w=5):
        idx = frame.index
        close = frame["close"].to_numpy()
        vals = []
        for ev in freezes:
            loc = idx.searchsorted(ev, side="left")
            if loc + 1 + w < len(idx):
                vals.append(np.log(close[loc + 1 + w] / close[loc + 1]))
        return float(np.mean(vals))

    assert post_freeze_mean(jump, tj["syn_freezes"]) > 0.05   # the planted spike shows up
    assert abs(post_freeze_mean(flat, tf["syn_freezes"])) < 0.03  # null ~ no effect


def test_winter_drift_tilts_winter_returns():
    rw, _ = data.synthetic_oj(winter_drift=0.001, seed=309)
    r = np.log(rw["close"]).diff().dropna()
    m = r.index.month
    winter = (m == 12) | (m == 1) | (m == 2)
    assert r[winter].mean() > r[~winter].mean()


def test_fetch_oj_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_oj("OJ=F", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    frame, _ = null_tape
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_oj(freeze_jump=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)
