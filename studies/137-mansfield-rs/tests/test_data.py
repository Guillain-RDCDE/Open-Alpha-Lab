"""The synthetic weekly tapes are well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mansfield_rs import data  # noqa: E402


def test_synthetic_keys(random_rs):
    """Tapes dict has SPY plus all synthetic stock tickers."""
    tapes, truth = random_rs
    assert "SPY" in tapes
    for t in truth["tickers"]:
        assert t in tapes


def test_synthetic_shape(random_rs):
    """Each tape has the expected number of weeks and OHLCV columns."""
    tapes, truth = random_rs
    for ticker, df in tapes.items():
        assert len(df) == truth["n_weeks"], f"{ticker}: expected {truth['n_weeks']} rows, got {len(df)}"
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}


def test_synthetic_ohlc_sanity(random_rs):
    """OHLC relationships hold: high >= open, close; low <= open, close; close > 0."""
    tapes, _ = random_rs
    for ticker, df in tapes.items():
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all(), ticker
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all(), ticker
        assert (df["high"] >= df["low"]).all(), ticker
        assert (df["close"] > 0).all(), ticker


def test_synthetic_is_deterministic():
    """Same seed gives identical tapes; different seed gives different tapes."""
    a, _ = data.synthetic_weekly(n_weeks=100, rs_momentum=0.1, seed=5)
    b, _ = data.synthetic_weekly(n_weeks=100, rs_momentum=0.1, seed=5)
    assert np.allclose(a["SPY"]["close"].to_numpy(), b["SPY"]["close"].to_numpy())
    c, _ = data.synthetic_weekly(n_weeks=100, rs_momentum=0.1, seed=6)
    assert not np.allclose(a["SPY"]["close"].to_numpy(), c["SPY"]["close"].to_numpy())


def test_rs_momentum_creates_persistence(trending_rs, random_rs):
    """Strong RS momentum creates serial dependence in excess returns; zero does not."""
    def autocorr(tapes, truth):
        bm = tapes["SPY"]["close"].to_numpy()
        acorrs = []
        for t in truth["tickers"][:5]:
            s = tapes[t]["close"].to_numpy()
            if len(bm) < 2 or len(s) < 2:
                continue
            n = min(len(s), len(bm))
            ex = np.log(s[:n] / s[:n].mean()) - np.log(bm[:n] / bm[:n].mean())
            r = np.diff(ex)
            if len(r) > 2:
                acorrs.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        return float(np.nanmean(acorrs)) if acorrs else 0.0

    tapes_t, truth_t = trending_rs
    tapes_r, truth_r = random_rs
    ac_trend = autocorr(tapes_t, truth_t)
    ac_flat = autocorr(tapes_r, truth_r)
    # Trending RS should show positive autocorrelation; flat should be near zero
    assert ac_trend > 0.05, f"Expected positive autocorr with RS momentum, got {ac_trend:.3f}"
    assert abs(ac_flat) < 0.15, f"Expected near-zero autocorr without RS momentum, got {ac_flat:.3f}"


def test_fetch_weekly_cache_only_raises_without_cache(tmp_path):
    """Requesting a ticker with no cache and fetch=False raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_weekly("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(random_rs):
    """Fingerprint is deterministic on the same tape and differs on a different tape."""
    tapes, _ = random_rs
    spy = tapes["SPY"]
    assert data.fingerprint(spy) == data.fingerprint(spy)
    other, _ = data.synthetic_weekly(n_weeks=260, rs_momentum=0.0, seed=99)
    assert data.fingerprint(spy) != data.fingerprint(other["SPY"])
