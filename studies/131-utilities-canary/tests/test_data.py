"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities_canary import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    expected_cols = {"XLU_close", "SPY_close", "XLU_ret", "SPY_ret", "rs", "rs_mom", "rs_mom_rank"}
    assert set(df.columns) >= expected_cols


def test_synthetic_prices_positive(null_tape):
    df, _ = null_tape
    assert (df["XLU_close"] > 0).all()
    assert (df["SPY_close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, canary_signal=0.0, seed=42)
    b, _ = data.synthetic_daily(n_days=500, canary_signal=0.0, seed=42)
    assert np.allclose(a["SPY_close"].to_numpy(), b["SPY_close"].to_numpy())
    # Different seed → different prices
    c, _ = data.synthetic_daily(n_days=500, canary_signal=0.0, seed=99)
    assert not np.allclose(a["SPY_close"].to_numpy(), c["SPY_close"].to_numpy())


def test_rs_is_log_ratio(null_tape):
    df, _ = null_tape
    expected_rs = np.log(df["XLU_close"]) - np.log(df["SPY_close"])
    assert np.allclose(df["rs"].to_numpy(), expected_rs.to_numpy(), equal_nan=True)


def test_rs_mom_is_20day_diff(null_tape):
    df, _ = null_tape
    expected = df["rs"].diff(20)
    assert np.allclose(
        df["rs_mom"].dropna().to_numpy(),
        expected.dropna().to_numpy(),
        atol=1e-12,
    )


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_days=6500, canary_signal=0.0, seed=999)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_canary_signal_changes_spy_returns():
    """Planted canary signal should shift average SPY returns in the alert quintile."""
    df_null, _ = data.synthetic_daily(n_days=3000, canary_signal=0.0, seed=131)
    df_cana, _ = data.synthetic_daily(n_days=3000, canary_signal=-0.01, seed=131)
    # With a planted negative signal, average SPY return should be different
    assert not np.allclose(
        df_null["SPY_ret"].to_numpy(),
        df_cana["SPY_ret"].to_numpy(),
    )
