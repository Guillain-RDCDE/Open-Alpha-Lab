"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_light import data  # noqa: E402


def test_synthetic_shape_and_ohlc(fair_open):
    bars, truth = fair_open
    assert len(bars) == truth["n_days"] * truth["bars_per_day"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range must bracket its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_5m(n_days=20, open_drift=10.0, seed=5)
    b, _ = data.synthetic_5m(n_days=20, open_drift=10.0, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_5m(n_days=20, open_drift=10.0, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_is_rth_only(fair_open):
    bars, _ = fair_open
    secs = bars.index.hour * 3600 + bars.index.minute * 60
    assert secs.min() >= 9 * 3600 + 30 * 60   # >= 09:30
    assert secs.max() <= 15 * 3600 + 55 * 60  # last bar opens 15:55


def test_null_drift_produces_no_systematic_open_bias():
    """With zero drift the first-bar return has no reliable direction across sessions."""
    bars, _ = data.synthetic_5m(n_days=200, open_drift=0.0, seed=73)
    session_date = bars.index.normalize()
    sessions = session_date.unique()
    first_bar_rets = []
    for day in sessions:
        day_bars = bars[session_date == day]
        if len(day_bars) >= 2:
            r = (day_bars["close"].iloc[0] - day_bars["open"].iloc[0]) / day_bars["open"].iloc[0]
            first_bar_rets.append(r)
    import numpy as np
    arr = np.array(first_bar_rets)
    # Mean first-bar return should be near zero with no drift planted.
    assert abs(arr.mean()) < 5e-4


def test_fetch_5m_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_5m("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(fair_open):
    bars, _ = fair_open
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_5m(n_days=60, open_drift=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
