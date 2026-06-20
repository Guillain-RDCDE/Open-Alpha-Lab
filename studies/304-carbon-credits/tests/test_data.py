"""The synthetic tape is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_credits import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    frame, truth = null_tape
    assert len(frame) == truth["n_days"]
    assert list(frame.columns) == ["KRBN", "CASH"]
    assert (frame["KRBN"] > 0).all()
    assert (frame["CASH"] > 0).all()
    assert frame.index.tz is None


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_carbon(n_days=400, mu=0.1, momentum=0.1, seed=7)
    b, _ = data.synthetic_carbon(n_days=400, mu=0.1, momentum=0.1, seed=7)
    assert np.allclose(a["KRBN"].to_numpy(), b["KRBN"].to_numpy())
    c, _ = data.synthetic_carbon(n_days=400, mu=0.1, momentum=0.1, seed=8)
    assert not np.allclose(a["KRBN"].to_numpy(), c["KRBN"].to_numpy())


def test_drift_lifts_terminal_level(null_tape, drift_tape):
    """A positive drift must compound to a higher terminal level than the null."""
    null, _ = null_tape
    drift, _ = drift_tape
    assert drift["KRBN"].iloc[-1] > null["KRBN"].iloc[-1]


def test_momentum_knob_creates_persistence():
    """The momentum knob really induces positive return autocorrelation (and 0 induces ~none)."""
    flat, _ = data.synthetic_carbon(n_days=4000, mu=0.0, momentum=0.0, seed=304)
    pers, _ = data.synthetic_carbon(n_days=4000, mu=0.0, momentum=0.40, seed=304)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    r_flat = np.diff(np.log(flat["KRBN"].to_numpy()))
    r_pers = np.diff(np.log(pers["KRBN"].to_numpy()))
    assert abs(ac(r_flat)) < 0.06
    assert ac(r_pers) > 0.25  # positive autocorrelation = momentum/persistence


def test_cash_leg_is_near_riskless(null_tape):
    """The CASH column should have negligible volatility vs the risky leg."""
    frame, _ = null_tape
    rk = np.diff(np.log(frame["KRBN"].to_numpy()))
    cs = np.diff(np.log(frame["CASH"].to_numpy()))
    assert cs.std() < rk.std() / 50.0


def test_load_real_cache_only_falls_back_or_loads(tmp_path):
    """With an empty cache dir and use_cache pointed at it, a miss tries the quantlab
    fallback — which without network raises; the cache path itself never raises on a hit."""
    # A non-existent ticker in an empty cache dir must not silently succeed.
    with pytest.raises(Exception):
        data.load_real(("NOPE_TICKER",), use_cache=True, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    frame, _ = null_tape
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_carbon(n_days=3000, mu=0.0, momentum=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# --- cache-gated real-tape test (skips offline) -----------------------------
_KRBN_CACHE = data._cache_path("KRBN", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_KRBN_CACHE), reason="offline CI: no real cache")
def test_real_tape_loads_from_cache():
    frame = data.load_real(("KRBN", "BIL"), end="2026-05-31")
    assert list(frame.columns) == ["KRBN", "BIL"]
    assert frame.index.is_monotonic_increasing
    assert (frame > 0).all().all()
