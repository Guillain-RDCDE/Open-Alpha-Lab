"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_seasonality import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: high brackets open and close from above.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, monthly_boost=0.1, seed=42)
    b, _ = data.synthetic_daily(n_years=5, monthly_boost=0.1, seed=42)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=5, monthly_boost=0.1, seed=43)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_boost_elevates_target_month(null_tape, uptober_tape):
    """Planting a large October boost increases the mean October log-return."""
    from crypto_seasonality.strategy import monthly_returns

    mret_null = monthly_returns(null_tape[0])
    mret_boost = monthly_returns(uptober_tape[0])

    oct_null = mret_null[mret_null["month"] == 10]["log_ret"].mean()
    oct_boost = mret_boost[mret_boost["month"] == 10]["log_ret"].mean()
    # Planted 0.50 log-return boost → boosted mean must be substantially larger.
    assert oct_boost > oct_null + 0.30


def test_boost_leaves_other_months_unchanged(null_tape, uptober_tape):
    """The planted boost is isolated to October; other months should be similar."""
    from crypto_seasonality.strategy import monthly_returns

    mret_null = monthly_returns(null_tape[0])
    mret_boost = monthly_returns(uptober_tape[0])

    # September mean should be close (same seed, only October differs structurally).
    sep_null = mret_null[mret_null["month"] == 9]["log_ret"].mean()
    sep_boost = mret_boost[mret_boost["month"] == 9]["log_ret"].mean()
    assert abs(sep_boost - sep_null) < 0.20  # same seed, small sample noise is fine


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    """Without a cache file, fetch_daily(fetch=False) raises FileNotFoundError."""
    # Temporarily override the study-83 fallback by patching the function.
    import importlib
    import unittest.mock as mock

    # Patch study83_cache to a nonexistent path as well.
    with mock.patch(
        "crypto_seasonality.data.os.path.exists", return_value=False
    ):
        with pytest.raises(FileNotFoundError):
            data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=11, monthly_boost=0.0, seed=999)
    assert data.fingerprint(bars) != data.fingerprint(other)
