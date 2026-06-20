"""The synthetic tapes are well-formed, deterministic, and carry the leading-digit law
we planted; the real loader is cache-safe (offline-gated)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benford_law import data  # noqa: E402

SHARED_SPY = os.path.join(data.SHARED_CACHE, "SPY_split_only.parquet")


# ---- synthetic tapes (never skip) -----------------------------------------
def test_benford_tape_shape_and_positive(benford_tape):
    s, truth = benford_tape
    assert len(s) == truth["n_days"]
    assert s.name == "close"
    assert (s > 0).all()
    assert np.isfinite(s.to_numpy()).all()


def test_benford_tape_spans_orders_of_magnitude(benford_tape):
    """The positive control must wander across >=2 decades — the Benford precondition."""
    _, truth = benford_tape
    assert truth["benford"] is True
    assert truth["log_range_decades"] >= 2.0


def test_nonbenford_tape_is_range_bound(nonbenford_tape):
    """The anti-control must stay inside its band (well under one decade of range)."""
    s, truth = nonbenford_tape
    assert truth["benford"] is False
    assert truth["log_range_decades"] < 0.5
    assert s.min() >= truth["center"] - truth["half_band"] - 1e-9
    assert s.max() <= truth["center"] + truth["half_band"] + 1e-9


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_benford(n_days=500, seed=7)
    b, _ = data.synthetic_benford(n_days=500, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_benford(n_days=500, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_decorative_index_is_timestamp_safe():
    """A long synthetic span must stay inside the ns-Timestamp range (no overflow)."""
    s, _ = data.synthetic_benford(n_days=12000, seed=1)
    assert s.index.is_monotonic_increasing
    assert s.index.max() < np.datetime64("2200-01-01")


def test_fingerprint_stable_and_sensitive(benford_tape):
    s, _ = benford_tape
    assert data.fingerprint(s) == data.fingerprint(s)
    other, _ = data.synthetic_benford(n_days=6000, seed=99)
    assert data.fingerprint(s) != data.fingerprint(other)


def test_load_real_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """With both caches pointed at an empty dir, a miss raises (no silent network)."""
    monkeypatch.setattr(data, "LOCAL_CACHE", str(tmp_path))
    monkeypatch.setattr(data, "SHARED_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_real("NOPE", fetch=False)


# ---- real tape (cache-gated: skips offline) -------------------------------
@pytest.mark.skipif(not os.path.exists(SHARED_SPY), reason="offline CI: shared cache absent")
def test_load_real_spy_from_shared_cache():
    s = data.load_real("SPY", mode="split_only")
    assert s.name == "close"
    assert (s > 0).all()
    assert getattr(s.index, "tz", None) is None
