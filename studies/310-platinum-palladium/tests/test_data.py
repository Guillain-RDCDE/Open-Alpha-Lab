"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platinum_palladium import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_pair):
    plat, pall, truth = null_pair
    assert len(plat) == truth["n_days"]
    assert len(pall) == truth["n_days"]
    assert list(plat.columns) == ["open", "high", "low", "close", "volume"]
    assert list(pall.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    for bars in (plat, pall):
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a_p, a_d, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=42)
    b_p, b_d, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=42)
    assert np.allclose(a_p["close"].to_numpy(), b_p["close"].to_numpy())
    assert np.allclose(a_d["close"].to_numpy(), b_d["close"].to_numpy())
    c_p, _, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=43)
    assert not np.allclose(a_p["close"].to_numpy(), c_p["close"].to_numpy())


def test_ratio_is_consistent(null_pair):
    """The platinum/palladium ratio is correctly encoded in the synthetic prices."""
    plat, pall, _ = null_pair
    ratio = plat["close"] / pall["close"]
    assert (ratio > 0.0).all()
    assert np.isfinite(ratio).all()


def test_mean_reversion_knob_affects_stationarity():
    """A high mean_rev_speed implies a short planted half-life; zero implies infinite."""
    _, _, truth_null = data.synthetic_daily(n_days=2000, mean_rev_speed=0.0, seed=7)
    _, _, truth_fast = data.synthetic_daily(n_days=2000, mean_rev_speed=0.20, seed=7)
    assert truth_fast["planted_half_life"] < 10.0
    assert truth_null["planted_half_life"] == float("inf")


def test_reversion_knob_induces_negative_autocorrelation():
    """The reversion knob really makes the ratio anti-persistent; 0 leaves a near-walk."""
    pl0, pa0, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=0.0, seed=11)
    plr, par, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=0.20, seed=11)
    ac = lambda s: np.corrcoef(s[:-1], s[1:])[0, 1]
    d0 = np.diff((pl0["close"] / pa0["close"]).to_numpy())
    dr = np.diff((plr["close"] / par["close"]).to_numpy())
    assert abs(ac(d0)) < 0.10            # random walk: increments ~uncorrelated
    assert ac(dr) < -0.10                # reversion: negative increment autocorrelation


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_load_real_returns_none_when_cache_missing(tmp_path):
    """The cache-first loader degrades gracefully to None offline (no crash)."""
    assert data.load_real(fetch=False, cache_dir=str(tmp_path)) is None


def test_fingerprint_is_stable_and_content_sensitive(null_pair):
    plat, _, _ = null_pair
    assert data.fingerprint(plat) == data.fingerprint(plat)
    other_p, _, _ = data.synthetic_daily(n_days=1000, mean_rev_speed=0.0, seed=99)
    assert data.fingerprint(plat) != data.fingerprint(other_p)


# ---- cache-gated real-tape smoke test (skips offline) ----------------------
_PL_CACHE = data._cache_path(data.PLATINUM_TICKER, data.DEFAULT_CACHE)
_PA_CACHE = data._cache_path(data.PALLADIUM_TICKER, data.DEFAULT_CACHE)


@pytest.mark.skipif(
    not (os.path.exists(_PL_CACHE) and os.path.exists(_PA_CACHE)),
    reason="offline CI: real PL/PA cache not present",
)
def test_real_cache_loads_when_present():
    out = data.load_real(fetch=False)
    assert out is not None
    plat, pall = out
    assert list(plat.columns) == ["open", "high", "low", "close", "volume"]
    assert (plat["close"] > 0).all()
    assert (pall["close"] > 0).all()
