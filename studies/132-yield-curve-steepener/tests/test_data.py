"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yield_curve_steepener import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert set(["slope", "TNX", "IRX", "TLT_close", "IEF_close", "SHY_close", "TLT_ret"]).issubset(
        set(df.columns)
    )


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, slope_signal=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=500, slope_signal=0.0, seed=5)
    assert np.allclose(a["TLT_close"].to_numpy(), b["TLT_close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=500, slope_signal=0.0, seed=6)
    assert not np.allclose(a["TLT_close"].to_numpy(), c["TLT_close"].to_numpy())


def test_synthetic_prices_positive(null_tape):
    df, _ = null_tape
    assert (df["TLT_close"] > 0).all()
    assert (df["IEF_close"] > 0).all()
    assert (df["SHY_close"] > 0).all()


def test_synthetic_yields_in_range(null_tape):
    df, _ = null_tape
    assert (df["TNX"] > 0).all()
    assert (df["IRX"] > 0).all()
    assert (df["TNX"] < 10.0).all()
    assert (df["IRX"] < 9.0).all()


def test_synthetic_slope_is_tnx_minus_irx(null_tape):
    df, _ = null_tape
    assert np.allclose(df["slope"].to_numpy(), (df["TNX"] - df["IRX"]).to_numpy())


def test_synthetic_index_is_business_days(null_tape):
    df, _ = null_tape
    # Business days only: no Saturdays or Sundays
    assert (df.index.dayofweek < 5).all()


def test_signal_knob_affects_tlt_ret_distribution():
    """A planted slope_signal > 0 makes TLT returns correlate with lagged slope rank."""
    null, _ = data.synthetic_daily(n_days=5000, slope_signal=0.0, seed=42)
    sig, _ = data.synthetic_daily(n_days=5000, slope_signal=0.01, seed=42)
    # When slope_signal > 0, high-slope-rank days should have higher TLT_ret on average.
    # Measure via the spread between top and bottom decile of slope rank.
    import pandas as pd
    for df, label in [(null, "null"), (sig, "signal")]:
        pct = df["slope"].rolling(252, min_periods=63).rank(pct=True)
        top = df.loc[pct > 0.8, "TLT_ret"].mean()
        bot = df.loc[pct < 0.2, "TLT_ret"].mean()
        spread = top - bot
        if label == "signal":
            assert spread > 0.0001, f"signal tape spread should be positive, got {spread}"
        else:
            assert abs(spread) < 0.001, f"null tape spread should be near zero, got {spread}"


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_days=3000, slope_signal=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)
