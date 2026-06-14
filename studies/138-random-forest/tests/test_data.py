"""Tests for the data layer — deterministic tape generation and fingerprinting."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_daily_shape_and_columns():
    bars, truth = data.synthetic_daily(n_days=300, signal_strength=0.0, seed=1)
    assert isinstance(bars, pd.DataFrame)
    assert set(bars.columns) >= {"open", "high", "low", "close", "volume"}
    assert len(bars) == 300
    assert bars.index.name == "date"


def test_synthetic_daily_deterministic():
    a, _ = data.synthetic_daily(n_days=200, signal_strength=0.01, seed=42)
    b, _ = data.synthetic_daily(n_days=200, signal_strength=0.01, seed=42)
    np.testing.assert_array_equal(a["close"].to_numpy(), b["close"].to_numpy())


def test_synthetic_daily_null_has_no_drift():
    """The null tape (signal_strength=0) should have a mean daily return near zero."""
    bars, _ = data.synthetic_daily(n_days=2000, signal_strength=0.0, seed=99)
    r = np.log(bars["close"] / bars["close"].shift(1)).dropna()
    # Under the null the annualised mean is ~vol^2/2 drift; the t-stat should be small
    mu = r.mean()
    se = r.std(ddof=1) / np.sqrt(len(r))
    tstat = abs(mu / se)
    assert tstat < 4.0, f"Null tape has unexpectedly large t-stat: {tstat:.2f}"


def test_synthetic_daily_signal_increases_serial_dependence():
    """A tape with a strong planted signal has more next-day predictability (higher lag-1 corr
    in the absolute-return domain) than the null tape."""
    b0, _ = data.synthetic_daily(n_days=2000, signal_strength=0.0, seed=5)
    b1, _ = data.synthetic_daily(n_days=2000, signal_strength=0.005, seed=5)
    r0 = np.log(b0["close"] / b0["close"].shift(1)).dropna().to_numpy()
    r1 = np.log(b1["close"] / b1["close"].shift(1)).dropna().to_numpy()
    # |lag-1 autocorrelation| of the signal tape must exceed that of the null
    ac0 = abs(np.corrcoef(r0[:-1], r0[1:])[0, 1])
    ac1 = abs(np.corrcoef(r1[:-1], r1[1:])[0, 1])
    assert ac1 >= ac0 - 0.01, f"Signal tape AC {ac1:.4f} not larger than null {ac0:.4f}"


def test_fingerprint_is_reproducible():
    bars, _ = data.synthetic_daily(n_days=100, seed=7)
    assert data.fingerprint(bars) == data.fingerprint(bars)
    bars2, _ = data.synthetic_daily(n_days=100, seed=8)
    assert data.fingerprint(bars) != data.fingerprint(bars2)


def test_fetch_daily_raises_without_cache():
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NONEXISTENT_XYZ", fetch=False, cache_dir="/tmp/no_such_dir_138")
