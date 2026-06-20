"""The synthetic tapes are well-formed, deterministic, and have the regime-character they
claim; the real loaders are cache-safe and never touch the network in CI."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uranium_revival import data  # noqa: E402


def test_synthetic_shape_and_ohlc(trending):
    bars, truth = trending
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=400, regime="trend", seed=5)
    b, _ = data.synthetic_daily(n_days=400, regime="trend", seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=400, regime="trend", seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        data.synthetic_daily(regime="moon-cycle")


def test_truth_flags_trendability():
    _, t_trend = data.synthetic_daily(n_days=400, regime="trend", seed=1)
    _, t_rand = data.synthetic_daily(n_days=400, regime="random", seed=1)
    _, t_hype = data.synthetic_daily(n_days=400, regime="hype", seed=1)
    assert t_trend["trendable"] is True
    assert t_rand["trendable"] is False
    assert t_hype["trendable"] is False


def test_trend_regime_has_direction_persistence(trending, coin):
    """The trend tape carries direction-persistence at the multi-day horizon a trend rule
    rides — 20-day momentum predicts the next 20-day move — while the random tape does not.
    (At lag 1 the daily noise dominates; persistence lives in the *regime*, not the day.)"""
    def mom_ac(bars, h=20):
        r = np.diff(np.log(bars["close"].to_numpy()))
        cum = pd.Series(r).rolling(h).sum().dropna().to_numpy()
        return np.corrcoef(cum[:-h], cum[h:])[0, 1]
    assert mom_ac(trending[0]) > 0.15        # persistent multi-day direction
    assert abs(mom_ac(coin[0])) < 0.15       # ~no persistence beyond overlap


def test_hype_regime_round_trips(hype):
    """The hype tape ramps up then crashes back: the peak is well above the end."""
    close = hype[0]["close"].to_numpy()
    peak = close.max()
    assert peak > close[0] * 2.0            # a real boom
    assert close[-1] < peak * 0.7           # it gave a lot back


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_load_real_raises_offline_without_cache(tmp_path):
    """With an empty cache dir and no shared cache hit, load_real raises (offline-safe)."""
    with pytest.raises(FileNotFoundError):
        data.load_real("DEFINITELY-NOT-A-TICKER-XYZ", cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(trending):
    bars, _ = trending
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=3000, regime="random", seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# --- cache-gated: only runs if a real parquet happens to be present locally ----
@pytest.mark.skipif(
    not os.path.exists(data._cache_path("URA")),
    reason="offline CI — no real URA cache parquet",
)
def test_real_ura_cache_is_well_formed():
    bars = data.load_real("URA")
    assert {"open", "high", "low", "close", "volume"} <= set(bars.columns)
    assert (bars["close"] > 0).all()
    assert bars.index.tz is None
