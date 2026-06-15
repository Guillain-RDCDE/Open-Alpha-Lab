"""Data-layer tests — shape, fingerprint, synthetic tape invariants, and cache gate.

Real-data tests are guarded with ``@requires_cache`` so they SKIP (not FAIL) in
offline CI where ``_cache/`` is absent.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_trend import data  # noqa: E402

_CACHE_PATH = data._cache_path(data.TICKER, data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="BTC-USD cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape invariants
# ---------------------------------------------------------------------------
def test_synthetic_shape(two_regime):
    prices, truth = two_regime
    expected_n = truth["n_years"] * data.TRADING_DAYS_PER_YEAR
    assert len(prices) == expected_n
    assert "close" in prices.columns
    assert "tbill" in prices.columns


def test_synthetic_index_is_daily(two_regime):
    prices, _ = two_regime
    diffs = pd.Series(prices.index).diff().dropna().dt.days
    # All consecutive days (including weekends — crypto never stops)
    assert (diffs == 1).all()


def test_synthetic_close_positive(two_regime):
    prices, _ = two_regime
    assert (prices["close"] > 0).all()


def test_synthetic_tbill_monotone(two_regime):
    prices, _ = two_regime
    assert (prices["tbill"].diff().dropna() > 0).all()


def test_synthetic_determinism():
    """Same seed must produce identical output every time."""
    p1, t1 = data.synthetic_daily(n_years=3, signal_strength=1.0, seed=42)
    p2, t2 = data.synthetic_daily(n_years=3, signal_strength=1.0, seed=42)
    assert np.allclose(p1["close"].to_numpy(), p2["close"].to_numpy())


def test_synthetic_null_vs_signal_strength():
    """Two-regime tape must have a bigger bull-bear vol spread than the null tape."""
    p_full, t_full = data.synthetic_daily(n_years=6, signal_strength=1.0, seed=210)
    p_null, t_null = data.synthetic_daily(n_years=6, signal_strength=0.0, seed=210)
    # The full-regime tape has higher bear effective vol — bear regime is more volatile
    assert t_full["vol_bear_eff"] > t_null["vol_bear_eff"]


def test_synthetic_truth_bear_frac(two_regime):
    prices, truth = two_regime
    # Bear fraction should be between 0 and 1 and roughly consistent with parameters
    assert 0.0 < truth["bear_frac"] < 1.0


def test_fingerprint_changes_with_data():
    p1, _ = data.synthetic_daily(n_years=3, signal_strength=1.0, seed=1)
    p2, _ = data.synthetic_daily(n_years=3, signal_strength=1.0, seed=2)
    assert data.fingerprint(p1) != data.fingerprint(p2)


def test_fingerprint_is_twelve_chars(two_regime):
    prices, _ = two_regime
    fp = data.fingerprint(prices)
    assert isinstance(fp, str) and len(fp) == 12


# ---------------------------------------------------------------------------
# Real-data tests (guarded — skip when cache absent)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_shape_and_start():
    df = data.fetch_prices(data.TICKER, fetch=False)
    assert "close" in df.columns
    assert len(df) > 3000       # BTC-USD has 4000+ days from 2014
    assert df.index[0].year == 2014


@requires_cache
def test_real_index_is_datetime():
    df = data.fetch_prices(data.TICKER, fetch=False)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"


@requires_cache
def test_real_close_positive():
    df = data.fetch_prices(data.TICKER, fetch=False)
    assert (df["close"] > 0).all()
    assert df["close"].notna().all()


@requires_cache
def test_real_fingerprint_stable():
    """Fingerprint should be reproducible for the same cached data."""
    df = data.fetch_prices(data.TICKER, fetch=False)
    fp1 = data.fingerprint(df)
    fp2 = data.fingerprint(df)
    assert fp1 == fp2
