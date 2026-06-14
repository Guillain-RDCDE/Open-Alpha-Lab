"""Synthetic tape is well-formed and deterministic; real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_powered_etf import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    prices, truth = null_tape
    assert len(prices) == truth["n_days"]
    assert list(prices.columns) == ["aieq", "spy", "ivv"]
    assert (prices > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=5)
    assert np.allclose(a["spy"].to_numpy(), b["spy"].to_numpy())
    c, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=6)
    assert not np.allclose(a["spy"].to_numpy(), c["spy"].to_numpy())


def test_synthetic_alpha_knob_shifts_return():
    """A higher planted alpha must produce a higher fund CAGR (market held fixed, seed fixed)."""
    p_low, _ = data.synthetic_daily(n_days=500, alpha_ann=-0.05, seed=7)
    p_high, _ = data.synthetic_daily(n_days=500, alpha_ann=0.05, seed=7)
    cagr_low = float((p_low["aieq"].iloc[-1] / p_low["aieq"].iloc[0]) ** (252 / 500) - 1)
    cagr_high = float((p_high["aieq"].iloc[-1] / p_high["aieq"].iloc[0]) ** (252 / 500) - 1)
    assert cagr_high > cagr_low


def test_synthetic_index_is_business_days(null_tape):
    prices, _ = null_tape
    diffs = np.diff(prices.index.values).astype("timedelta64[D]").astype(int)
    # Business days: gaps should only be 1, 2, or 3 days (weekend), never 0
    assert diffs.min() >= 1
    assert diffs.max() <= 3  # at most long weekends


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    prices, _ = null_tape
    assert data.fingerprint(prices["spy"]) == data.fingerprint(prices["spy"])
    other, _ = data.synthetic_daily(n_days=600, alpha_ann=0.0, seed=99)
    assert data.fingerprint(prices["spy"]) != data.fingerprint(other["spy"])


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(tickers=["AIEQ"], fetch=False, cache_dir=str(tmp_path))
