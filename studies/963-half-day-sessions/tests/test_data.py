"""Data-layer tests for Study 963 — synthetic determinism offline, cache-gated on tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_day import data  # noqa: E402


def test_synthetic_ohlc_is_deterministic():
    a, _ = data.synthetic_ohlc(n_years=4, seed=963)
    b, _ = data.synthetic_ohlc(n_years=4, seed=963)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_ohlc_seed_sensitive():
    a, _ = data.synthetic_ohlc(n_years=4, seed=963)
    b, _ = data.synthetic_ohlc(n_years=4, seed=963 + 1)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_bars_are_internally_consistent():
    bars, truth = data.synthetic_ohlc(n_years=5, seed=963)
    assert list(bars.columns) == list(data.BAR_COLS)
    assert (bars["volume"] > 0).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (bars > 0).all().all()
    assert len(bars) == truth["n_days"]
    assert bars.index[-1] < pd.Timestamp("2262-01-01")


def test_range_is_never_degenerate():
    """A zero range would make every range estimator undefined — the generator must not."""
    bars, _ = data.synthetic_ohlc(n_years=5, seed=963)
    assert (np.log(bars["high"] / bars["low"]) > 0).all()


def test_signal_strength_switches_clustering_off():
    _, t1 = data.synthetic_ohlc(n_years=8, signal_strength=1.0, seed=963)
    _, t0 = data.synthetic_ohlc(n_years=8, signal_strength=0.0, seed=963)
    assert t1["sigma"].std() > 0
    assert np.isclose(t0["sigma"].std(), 0.0, atol=1e-15)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_ohlc(n_years=3, seed=963)
    b, _ = data.synthetic_ohlc(n_years=3, seed=963 + 1)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_ohlc_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_ohlc(data.TICKERS[0], cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_universe_is_declared():
    assert len(data.TICKERS) == len(set(data.TICKERS)) >= 1
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.START)


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_bars_load_sane_and_pinned():
    bars = data.load_ohlc(data.TICKERS[0])
    assert bars.index[-1] <= pd.Timestamp(data.AS_OF)
    assert (bars["high"] >= bars["low"]).all()
    assert bars.index.is_monotonic_increasing and not bars.index.has_duplicates
