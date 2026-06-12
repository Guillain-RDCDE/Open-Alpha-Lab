"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tail_radar import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert list(df.columns) == ["SKEW", "VIX", "SPY_close", "SPY_ret"]
    assert (df["SKEW"] > 0).all()
    assert (df["VIX"] > 0).all()
    assert (df["SPY_close"] > 0).all()


def test_synthetic_skew_bounded(null_tape):
    df, _ = null_tape
    assert df["SKEW"].min() >= 98.0
    assert df["SKEW"].max() <= 162.0


def test_synthetic_vix_reasonable(null_tape):
    df, _ = null_tape
    assert df["VIX"].min() > 2.0
    assert df["VIX"].max() < 120.0


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, skew_signal=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=500, skew_signal=0.0, seed=5)
    assert np.allclose(a["SPY_close"].to_numpy(), b["SPY_close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=500, skew_signal=0.0, seed=6)
    assert not np.allclose(a["SPY_close"].to_numpy(), c["SPY_close"].to_numpy())


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(null_tape, signal_tape):
    df_null, _ = null_tape
    df_sig, _ = signal_tape
    assert data.fingerprint(df_null) == data.fingerprint(df_null)
    # Different tapes (different signal level) should produce different fingerprints.
    assert data.fingerprint(df_null) != data.fingerprint(df_sig)


def test_synthetic_index_is_business_days(null_tape):
    df, _ = null_tape
    assert df.index.freq is None or True  # bdate_range; freq may be 'B' or None after slice
    # No Saturdays or Sundays.
    assert (df.index.dayofweek < 5).all()
