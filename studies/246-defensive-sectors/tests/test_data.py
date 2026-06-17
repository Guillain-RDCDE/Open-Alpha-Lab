"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defensive_sectors import data  # noqa: E402

CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "daily_xlp_xlu_spy.parquet")
)


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    expected_cols = {
        "XLP_close", "XLU_close", "SPY_close",
        "XLP_ret", "XLU_ret", "SPY_ret",
        "combined_rs", "rs_mom", "rs_mom_rank",
    }
    assert set(df.columns) >= expected_cols


def test_synthetic_prices_positive(null_tape):
    df, _ = null_tape
    assert (df["XLP_close"] > 0).all()
    assert (df["XLU_close"] > 0).all()
    assert (df["SPY_close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, defensive_signal=0.0, seed=42)
    b, _ = data.synthetic_daily(n_days=500, defensive_signal=0.0, seed=42)
    assert np.allclose(a["SPY_close"].to_numpy(), b["SPY_close"].to_numpy())
    # Different seed -> different prices
    c, _ = data.synthetic_daily(n_days=500, defensive_signal=0.0, seed=99)
    assert not np.allclose(a["SPY_close"].to_numpy(), c["SPY_close"].to_numpy())


def test_combined_rs_is_average_of_log_ratios(null_tape):
    df, _ = null_tape
    import numpy as np
    rs_xlp = np.log(df["XLP_close"]) - np.log(df["SPY_close"])
    rs_xlu = np.log(df["XLU_close"]) - np.log(df["SPY_close"])
    expected = 0.5 * (rs_xlp + rs_xlu)
    assert np.allclose(df["combined_rs"].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_rs_mom_is_20day_diff(null_tape):
    df, _ = null_tape
    expected = df["combined_rs"].diff(20)
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
    other, _ = data.synthetic_daily(n_days=6500, defensive_signal=0.0, seed=999)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_defensive_signal_changes_spy_returns():
    """Planted defensive signal should shift average SPY returns in the alert quintile."""
    df_null, _ = data.synthetic_daily(n_days=3000, defensive_signal=0.0, seed=246)
    df_def, _ = data.synthetic_daily(n_days=3000, defensive_signal=-0.01, seed=246)
    assert not np.allclose(
        df_null["SPY_ret"].to_numpy(),
        df_def["SPY_ret"].to_numpy(),
    )


@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_loads_and_has_three_tickers():
    df = data.fetch_daily(fetch=False)
    for col in ("XLP_close", "XLU_close", "SPY_close"):
        assert col in df.columns
    assert len(df) > 5000
