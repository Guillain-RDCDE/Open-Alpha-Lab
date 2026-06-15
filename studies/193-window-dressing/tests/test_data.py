"""The synthetic tape is well-formed and deterministic; label logic is correct;
the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from window_dressing import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for real-data tests
# ---------------------------------------------------------------------------
_SPY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "SPY_total_return.parquet"
)
_GSPC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "^GSPC_split_only.parquet"
)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_SPY_PATH) or os.path.exists(_GSPC_PATH)),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"ret", "is_pump", "is_reversal", "seq_rev", "seq_fwd"}
    assert df["ret"].dtype == float


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, wd_effect=0.5, seed=7)
    b, _ = data.synthetic_daily(n_years=10, wd_effect=0.5, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, wd_effect=0.5, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_synthetic_no_nans_in_ret(null_tape):
    df, _ = null_tape
    assert df["ret"].isna().sum() == 0


def test_pump_and_reversal_are_disjoint(null_tape):
    """A day cannot be both a pump day and a reversal day."""
    df, _ = null_tape
    assert not (df["is_pump"] & df["is_reversal"]).any()


def test_pump_count_reasonable(null_tape):
    """30 years × 4 quarter-ends × 5 days = 600 pump days (approximately)."""
    df, truth = null_tape
    assert 500 <= truth["n_pump"] <= 700


def test_reversal_count_reasonable(null_tape):
    """Approximately symmetric: ~600 reversal days over 30 years."""
    df, truth = null_tape
    assert 500 <= truth["n_reversal"] <= 700


def test_seq_rev_within_window(null_tape):
    """seq_rev for pump days is between 1 and window."""
    df, truth = null_tape
    pump_days = df[df["is_pump"]]
    assert (pump_days["seq_rev"] >= 1).all()
    assert (pump_days["seq_rev"] <= data.DEFAULT_WINDOW).all()


def test_seq_fwd_within_window(null_tape):
    """seq_fwd for reversal days is between 1 and window."""
    df, truth = null_tape
    rev_days = df[df["is_reversal"]]
    assert (rev_days["seq_fwd"] >= 1).all()
    assert (rev_days["seq_fwd"] <= data.DEFAULT_WINDOW).all()


def test_effect_plants_pump_premium():
    """With a strong planted effect, pump days have a higher mean return than others."""
    df, _ = data.synthetic_daily(n_years=50, wd_effect=2.0, seed=193)
    pump_mean = df.loc[df["is_pump"], "ret"].mean()
    other_mean = df.loc[~df["is_pump"] & ~df["is_reversal"], "ret"].mean()
    assert pump_mean > other_mean


def test_effect_plants_reversal_depression():
    """With a strong planted effect, reversal days have a lower mean return than others."""
    df, _ = data.synthetic_daily(n_years=50, wd_effect=2.0, seed=193)
    rev_mean = df.loc[df["is_reversal"], "ret"].mean()
    other_mean = df.loc[~df["is_pump"] & ~df["is_reversal"], "ret"].mean()
    assert rev_mean < other_mean


def test_null_tape_has_no_systematic_pump_premium(null_tape):
    """On a null tape the pump-day mean should be close to the baseline (no planted signal)."""
    df, _ = null_tape
    pump_mean = df.loc[df["is_pump"], "ret"].mean()
    other_mean = df.loc[~df["is_pump"], "ret"].mean()
    # With no effect planted the contrast should be small relative to daily vol ~0.01.
    assert abs(pump_mean - other_mean) < 0.002


def test_fingerprint_stable_and_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_years=30, wd_effect=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_fetch_raises_without_cache(tmp_path):
    """fetch_spy raises when no cache is available and fetch=False."""
    import importlib, sys as _sys
    # Temporarily redirect cache paths to a tmp dir with no files.
    orig_spy = data._SPY_SHARED
    orig_gspc = data._GSPC_SHARED
    data._SPY_SHARED = str(tmp_path / "nope_spy.parquet")
    data._GSPC_SHARED = str(tmp_path / "nope_gspc.parquet")
    try:
        with pytest.raises(FileNotFoundError):
            data.fetch_spy(fetch=False, cache_dir=str(tmp_path))
    finally:
        data._SPY_SHARED = orig_spy
        data._GSPC_SHARED = orig_gspc


# ---------------------------------------------------------------------------
# Real-data tests (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_tape_shape():
    df = data.fetch_spy()
    assert len(df) > 1000
    assert set(df.columns) >= {"ret", "is_pump", "is_reversal"}
    assert df["ret"].isna().sum() == 0


@requires_cache
def test_real_tape_labels_sane():
    df = data.fetch_spy()
    # Pump and reversal are disjoint on the real tape too.
    assert not (df["is_pump"] & df["is_reversal"]).any()
    # At least 100 pump events (>25 years × 4 × 5 > 100).
    assert df["is_pump"].sum() >= 100
    assert df["is_reversal"].sum() >= 100
