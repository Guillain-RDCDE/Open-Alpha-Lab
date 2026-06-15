"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_rainbow import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert set(["close", "log_close", "log_days", "band_sigma_insample"]).issubset(df.columns)
    assert (df["close"] > 0).all()
    assert np.allclose(df["log_close"].to_numpy(), np.log(df["close"].to_numpy()))


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_btc(n_days=200, signal_strength=0.5, seed=10)
    b, _ = data.synthetic_btc(n_days=200, signal_strength=0.5, seed=10)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_btc(n_days=200, signal_strength=0.5, seed=11)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_log_days_is_positive_and_increasing():
    import pandas as pd
    dates = pd.bdate_range("2014-01-01", periods=500, name="date")
    ld = data.log_days_since_genesis(dates)
    assert (ld > 0).all()
    assert (ld.diff().dropna() >= 0).all(), "log_days should be non-decreasing"


def test_log_days_genesis_floor():
    """Days at or before genesis should be floored to log(1) = 0."""
    import pandas as pd
    dates = pd.DatetimeIndex([pd.Timestamp("2009-01-03"), pd.Timestamp("2009-01-04")])
    ld = data.log_days_since_genesis(dates)
    # Genesis = day 0 floored to 1 → log(1) = 0
    assert ld.iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert ld.iloc[1] == pytest.approx(np.log(1.0), abs=1e-6)


def test_signal_tape_has_trend(signal_tape):
    """When signal_strength=1, log_close should be positively correlated with log_days.
    The noise is a random walk so the correlation won't be perfect, but should be
    clearly positive and stronger than the null tape."""
    df, _ = signal_tape
    corr = df[["log_close", "log_days"]].corr().iloc[0, 1]
    assert corr > 0.3, f"Expected positive correlation on signal tape; got {corr:.2f}"


def test_null_tape_weaker_trend(null_tape, signal_tape):
    """The null tape (random walk) should have a weaker log_days correlation than the signal tape."""
    df_null, _ = null_tape
    df_sig, _ = signal_tape
    c_null = df_null[["log_close", "log_days"]].corr().iloc[0, 1]
    c_sig = df_sig[["log_close", "log_days"]].corr().iloc[0, 1]
    assert abs(c_null) < abs(c_sig), (
        f"Expected null tape corr ({c_null:.2f}) < signal tape corr ({c_sig:.2f})"
    )


def test_fetch_btc_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """Without any cached file (local or shared), fetch_btc should raise FileNotFoundError."""
    # Monkeypatch the shared-cache lookup so it also points to the empty tmp_path.
    monkeypatch.setattr(data, "_shared_cache_path", lambda: str(tmp_path / "nonexistent.parquet"))
    with pytest.raises(FileNotFoundError):
        data.fetch_btc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_btc(n_days=1_500, signal_strength=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)
