"""Data-layer invariants for Study 342 (BDC-Yield)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bdc_yield import data  # noqa: E402


def test_shape_and_columns(equity_like):
    frame, truth = equity_like
    assert list(frame.columns) == ["BIZD", "SPY", "BND"]
    assert len(frame) == truth["n_days"] == 3300
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_three_asset(n_days=1000, bdc_beta=1.1, seed=7)
    b, _ = data.synthetic_three_asset(n_days=1000, bdc_beta=1.1, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_bdc_beta_drives_equity_correlation(equity_like, bond_like):
    # High bdc_beta => BIZD tracks stocks more than bonds; low => the reverse.
    _, hi = equity_like
    _, lo = bond_like
    assert hi["corr_bizd_stk"] > hi["corr_bizd_bnd"]
    assert lo["corr_bizd_bnd"] > lo["corr_bizd_stk"]


def test_crash_window_present(equity_like):
    _, truth = equity_like
    assert truth["crash_trough"] > truth["crash_peak"]


def test_return_of_capital_recorded(equity_like):
    _, truth = equity_like
    assert truth["distribution"] > truth["return_of_capital"] > 0


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_three_asset(n_days=500, bdc_beta=1.1, seed=1)
    b, _ = data.synthetic_three_asset(n_days=500, bdc_beta=1.1, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_changes_with_columns():
    a, _ = data.synthetic_three_asset(n_days=500, bdc_beta=1.1, seed=1)
    b = a.rename(columns={"BIZD": "BDCX"})
    assert data.fingerprint(a) != data.fingerprint(b)


# --- real cache, GATED: only runs if a cached tape is actually present ---
import pytest  # noqa: E402

_SHARED = os.path.join(data.SHARED_CACHE, "SPY_total_return.parquet")
_LOCAL = os.path.join(data.LOCAL_CACHE, "BIZD_total_return.parquet")


@pytest.mark.skipif(
    not (os.path.exists(_SHARED) and os.path.exists(_LOCAL)),
    reason="offline CI — real BIZD/SPY caches absent",
)
def test_load_real_shape():
    frame = data.load_real(("BIZD", "SPY", "IEF"))
    assert list(frame.columns) == ["BIZD", "SPY", "IEF"]
    assert len(frame) > 100
    assert (frame > 0).all().all()
