"""Data layer tests — synthetic tape shape, determinism, MM computation, cache safety."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mayer_multiple import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    for col in ["close", "log_return", "sma_200", "mayer_multiple"]:
        assert col in df.columns, f"Missing column: {col}"
    assert (df["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_btc(n_days=300, signal_strength=0.5, seed=10)
    b, _ = data.synthetic_btc(n_days=300, signal_strength=0.5, seed=10)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_btc(n_days=300, signal_strength=0.5, seed=11)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_sma_warmup_is_nan(null_tape):
    df, _ = null_tape
    # First SMA_WINDOW - 1 rows should be NaN for the SMA.
    assert df["sma_200"].iloc[: data.SMA_WINDOW - 1].isna().all()
    # After the warm-up there should be valid SMA values.
    assert df["sma_200"].iloc[data.SMA_WINDOW :].notna().sum() > 0


def test_mayer_multiple_positive(null_tape):
    df, _ = null_tape
    mm = df["mayer_multiple"].dropna()
    assert (mm > 0).all(), "Mayer Multiple must be strictly positive"


def test_mayer_multiple_equals_close_over_sma(null_tape):
    df, _ = null_tape
    valid = df["sma_200"].notna()
    computed = df.loc[valid, "close"] / df.loc[valid, "sma_200"]
    np.testing.assert_allclose(
        df.loc[valid, "mayer_multiple"].to_numpy(),
        computed.to_numpy(),
        rtol=1e-9,
    )


def test_fetch_btc_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """Without any cached file the function should raise FileNotFoundError."""
    monkeypatch.setattr(data, "_shared_cache_path", lambda: str(tmp_path / "nonexistent.parquet"))
    with pytest.raises(FileNotFoundError):
        data.fetch_btc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive(null_tape):
    df, _ = null_tape
    fp1 = data.fingerprint(df)
    fp2 = data.fingerprint(df)
    assert fp1 == fp2
    other, _ = data.synthetic_btc(n_days=2_000, signal_strength=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Cache-gated real-tape test (skipped offline / CI)
# ---------------------------------------------------------------------------
_LOCAL_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "btc_usd_daily.parquet")
)
_SHARED_CACHE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "174-bitcoin-rainbow", "_cache", "btc_usd_daily.parquet"
    )
)


@pytest.mark.skipif(
    not (os.path.exists(_LOCAL_CACHE) or os.path.exists(_SHARED_CACHE)),
    reason="real-tape cache absent offline/CI",
)
def test_real_tape_has_correct_columns():
    df = data.fetch_btc(fetch=False)
    for col in ["close", "log_return", "sma_200", "mayer_multiple"]:
        assert col in df.columns
    assert len(df) > 3_000, "Expected multi-year BTC history"
