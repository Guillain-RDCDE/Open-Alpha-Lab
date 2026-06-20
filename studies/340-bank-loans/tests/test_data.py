"""Data-layer invariants for Study 340 (Bank-Loans)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bank_loans import data  # noqa: E402


def test_shape_and_columns(floating_like):
    frame, truth = floating_like
    assert list(frame.columns) == ["BKLN", "TLT", "IEF", "SPY"]
    assert len(frame) == truth["n_days"] == 3700
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_four_asset(n_days=1000, dur=0.1, credit_beta=0.5, seed=7)
    b, _ = data.synthetic_four_asset(n_days=1000, dur=0.1, credit_beta=0.5, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_dur_drives_rate_correlation(floating_like, duration_like):
    # Low dur => loan barely correlates with the rate factor; high dur => strongly (negative).
    _, flo = floating_like
    _, dur = duration_like
    assert abs(flo["corr_loan_rate"]) < abs(dur["corr_loan_rate"])
    # The duration bond loads negatively on the rate level (price falls as rates rise).
    assert dur["corr_loan_rate"] < 0


def test_credit_beta_drives_equity_correlation(floating_like, duration_like):
    # High credit_beta => loan tracks stocks; low => it doesn't.
    _, flo = floating_like
    _, dur = duration_like
    assert flo["corr_loan_stk"] > dur["corr_loan_stk"]
    assert flo["corr_loan_stk"] > 0.3


def test_windows_present(floating_like):
    _, truth = floating_like
    assert truth["rate_spike_end"] > truth["rate_spike_peak"]
    assert truth["crash_trough"] > truth["crash_peak"]


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_four_asset(n_days=500, dur=0.1, credit_beta=0.5, seed=1)
    b, _ = data.synthetic_four_asset(n_days=500, dur=0.1, credit_beta=0.5, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_changes_with_columns():
    a, _ = data.synthetic_four_asset(n_days=500, dur=0.1, credit_beta=0.5, seed=1)
    b = a.rename(columns={"BKLN": "SRLN"})
    assert data.fingerprint(a) != data.fingerprint(b)


def test_short_tape_no_index_error():
    # crash/spike windows clamp safely on a tape shorter than the default window starts.
    frame, truth = data.synthetic_four_asset(n_days=200, dur=0.1, credit_beta=0.5, seed=3)
    assert len(frame) == 200
    assert (frame > 0).all().all()


# --- real cache, GATED: only runs if cached tapes are actually present ---
import pytest  # noqa: E402

_SHARED = os.path.join(data.SHARED_CACHE, "TLT_total_return.parquet")
_LOCAL = os.path.join(data.LOCAL_CACHE, "BKLN_total_return.parquet")


@pytest.mark.skipif(
    not (os.path.exists(_SHARED) and os.path.exists(_LOCAL)),
    reason="offline CI — real BKLN/TLT caches absent",
)
def test_load_real_shape():
    frame = data.load_real(("BKLN", "TLT", "IEF", "SPY"))
    assert list(frame.columns) == ["BKLN", "TLT", "IEF", "SPY"]
    assert len(frame) > 100
    assert (frame > 0).all().all()
