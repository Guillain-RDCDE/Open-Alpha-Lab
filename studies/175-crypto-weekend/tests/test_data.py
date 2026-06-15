"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_weekend import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, weekend_boost=0.0, seed=42)
    b, _ = data.synthetic_daily(n_years=5, weekend_boost=0.0, seed=42)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=5, weekend_boost=0.0, seed=43)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_covers_all_seven_days(null_tape):
    """BTC trades 7 days a week: the index must include weekends."""
    bars, _ = null_tape
    dows = set(bars.index.dayofweek.unique())
    assert 5 in dows  # Saturday
    assert 6 in dows  # Sunday
    assert len(dows) == 7


def test_weekend_boost_shifts_weekend_mean(null_tape, boosted_tape):
    """The boost knob reliably raises the mean weekend return relative to weekdays."""
    import pandas as pd
    from crypto_weekend import strategy as st

    for tape, _ in [null_tape, boosted_tape]:
        d = st.daily_returns(tape)
        assert "is_weekend" in d.columns
        assert d["is_weekend"].any()
        assert (~d["is_weekend"]).any()

    d_null = st.daily_returns(null_tape[0])
    d_boost = st.daily_returns(boosted_tape[0])

    null_premium = (
        d_null[d_null["is_weekend"]]["log_ret"].mean()
        - d_null[~d_null["is_weekend"]]["log_ret"].mean()
    )
    boost_premium = (
        d_boost[d_boost["is_weekend"]]["log_ret"].mean()
        - d_boost[~d_boost["is_weekend"]]["log_ret"].mean()
    )
    assert boost_premium > null_premium + 0.01


def test_truth_records_correct_day_counts(null_tape):
    bars, truth = null_tape
    n_sat_sun = int((bars.index.dayofweek >= 5).sum())
    n_weekdays = int((bars.index.dayofweek < 5).sum())
    assert truth["n_weekends"] == n_sat_sun
    assert truth["n_weekdays"] == n_weekdays


def test_fetch_daily_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """Without a local cache and without the study-133 fallback, must raise."""
    # Patch the fallback path so neither cache is found.
    monkeypatch.setattr(data, "_STUDY_133_CACHE", str(tmp_path / "nonexistent.parquet"))
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=8, weekend_boost=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
