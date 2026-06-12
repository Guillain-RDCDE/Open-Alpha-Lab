"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind_the_gap import data  # noqa: E402


def test_synthetic_shape_and_ohlc(coin):
    daily, truth = coin
    assert len(daily) == truth["n_days"] - 1  # first row dropped (no prev_close)
    expected_cols = {"open", "high", "low", "close", "volume", "gap_pct", "gap_fill"}
    assert expected_cols.issubset(set(daily.columns))
    # OHLC sanity: range brackets open and close.
    assert (daily["high"] >= daily[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (daily["low"]  <= daily[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (daily["high"] >= daily["low"]).all()
    assert (daily["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=100, gap_fill_prob=0.5, seed=5)
    b, _ = data.synthetic_daily(n_days=100, gap_fill_prob=0.5, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=100, gap_fill_prob=0.5, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_gap_pct_signed_correctly(coin):
    daily, _ = coin
    # For days where open > prev_close, gap_pct must be positive.
    prev_close = daily["close"].shift(1)
    up_gaps = daily[daily["open"] > prev_close.shift(-1 * 0)]  # proxy: just check sign consistency
    # gap_pct = (open - prev_close) / prev_close * 100
    # Manually recompute for the first few rows and check.
    for i in range(5, min(15, len(daily))):
        row = daily.iloc[i]
        prev = daily["close"].iloc[i - 1]
        expected = (row["open"] - prev) / prev * 100.0
        assert abs(row["gap_pct"] - expected) < 1e-6, f"gap_pct mismatch at row {i}"


def test_gap_fill_flag_logic(coin):
    """gap_fill is True iff the day's range includes the prior close."""
    daily, _ = coin
    prev_close = daily["close"].shift(1).reindex(daily.index)
    for i in range(1, min(20, len(daily))):
        row = daily.iloc[i]
        pc  = daily["close"].iloc[i - 1]
        gp  = row["gap_pct"]
        if gp > 0:
            expected = row["low"] <= pc
        elif gp < 0:
            expected = row["high"] >= pc
        else:
            expected = True  # zero gap always 'filled'
        assert row["gap_fill"] == expected, f"gap_fill mismatch at row {i}"


def test_high_fill_prob_raises_fill_rate(mean_reverting, coin):
    """A higher baked-in gap_fill_prob actually produces more fills."""
    daily_coin, _ = coin
    daily_rev,  _ = mean_reverting
    rate_coin = daily_coin["gap_fill"].mean()
    rate_rev  = daily_rev["gap_fill"].mean()
    assert rate_rev > rate_coin, (
        f"mean_reverting fill rate {rate_rev:.3f} should exceed coin {rate_coin:.3f}"
    )


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin):
    daily, _ = coin
    assert data.fingerprint(daily) == data.fingerprint(daily)
    other, _ = data.synthetic_daily(n_days=500, gap_fill_prob=0.5, seed=99)
    assert data.fingerprint(daily) != data.fingerprint(other)
