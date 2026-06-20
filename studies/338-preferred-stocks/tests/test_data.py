"""Data-layer invariants for Study 338 (Preferred-Stocks)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preferred_stocks import data  # noqa: E402


def test_shape_and_columns(equity_like):
    frame, truth = equity_like
    assert list(frame.columns) == ["PFF", "SPY", "BND"]
    assert len(frame) == truth["n_days"] == 4500
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_three_asset(n_days=1000, pref_beta=0.8, seed=7)
    b, _ = data.synthetic_three_asset(n_days=1000, pref_beta=0.8, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_pref_beta_drives_equity_correlation(equity_like, bond_like):
    # High pref_beta => PFF tracks stocks more than bonds; low => the reverse.
    _, hi = equity_like
    _, lo = bond_like
    assert hi["corr_pff_stk"] > hi["corr_pff_bnd"]
    assert lo["corr_pff_bnd"] > lo["corr_pff_stk"]


def test_crash_window_present(equity_like):
    _, truth = equity_like
    assert truth["crash_trough"] > truth["crash_peak"]


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_three_asset(n_days=500, pref_beta=0.8, seed=1)
    b, _ = data.synthetic_three_asset(n_days=500, pref_beta=0.8, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_changes_with_columns():
    a, _ = data.synthetic_three_asset(n_days=500, pref_beta=0.8, seed=1)
    b = a.rename(columns={"PFF": "PGX"})
    assert data.fingerprint(a) != data.fingerprint(b)


# --- real cache, GATED: only runs if a cached tape is actually present ---
import pytest  # noqa: E402

_SHARED = os.path.join(data.SHARED_CACHE, "SPY_total_return.parquet")
_LOCAL = os.path.join(data.LOCAL_CACHE, "PFF_total_return.parquet")


@pytest.mark.skipif(
    not (os.path.exists(_SHARED) and os.path.exists(_LOCAL)),
    reason="offline CI — real PFF/SPY caches absent",
)
def test_load_real_shape():
    frame = data.load_real(("PFF", "SPY", "IEF"))
    assert list(frame.columns) == ["PFF", "SPY", "IEF"]
    assert len(frame) > 100
    assert (frame > 0).all().all()
