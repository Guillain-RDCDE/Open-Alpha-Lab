"""The synthetic tape is well-formed, deterministic, and honest about its ground truth."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algo_overfit import data  # noqa: E402

FEATS = data.FEATURE_NAMES


def test_shape_and_columns(null_tape):
    assert len(null_tape) == 1499  # n_days - 1 (last row has no forward return)
    for c in ["price", "ret", "fwd_ret", *FEATS]:
        assert c in null_tape.columns
    assert (null_tape["price"] > 0).all()
    assert np.isfinite(null_tape[FEATS].to_numpy()).all()


def test_determinism():
    a, _ = data.synthetic_series(n_days=800, signal_strength=0.2, seed=7)
    b, _ = data.synthetic_series(n_days=800, signal_strength=0.2, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    c, _ = data.synthetic_series(n_days=800, signal_strength=0.2, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy(), equal_nan=True)


def test_null_has_no_edge_control_does():
    _, t0 = data.synthetic_series(n_days=500, signal_strength=0.0, seed=1)
    _, t1 = data.synthetic_series(n_days=500, signal_strength=0.3, seed=1)
    assert not t0.has_edge
    assert t1.has_edge


def test_fwd_ret_is_next_day_return(null_tape):
    """fwd_ret[t] must equal ret[t+1] — the one execution lag, applied in the data layer."""
    r = null_tape["ret"].to_numpy()
    f = null_tape["fwd_ret"].to_numpy()
    assert np.allclose(f[:-1], r[1:][: len(f) - 1], atol=1e-12)


def test_features_are_standardised(null_tape):
    for c in FEATS:
        s = null_tape[c]
        assert abs(s.mean()) < 0.2          # roughly centred
        assert 0.5 < s.std(ddof=0) < 2.0    # roughly unit scale


def test_fingerprint_stable_and_sensitive(null_tape):
    fp = data.fingerprint(null_tape)
    assert isinstance(fp, str) and len(fp) == 12
    other, _ = data.synthetic_series(n_days=1500, signal_strength=0.3, seed=589)
    assert data.fingerprint(other) != fp
