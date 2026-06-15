"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hot_hand import data  # noqa: E402


def test_synthetic_shape(coin):
    bars, truth = coin
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["close"]
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, autocorr=0.1, seed=5)
    b, _ = data.synthetic_daily(n_days=200, autocorr=0.1, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, autocorr=0.1, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_synthetic_has_business_day_index(coin):
    bars, _ = coin
    # Check index has no weekends
    assert (bars.index.dayofweek < 5).all()


def test_autocorr_knob_works():
    """The autocorr parameter induces the correct sign of serial dependence."""
    def lag1_ac(bars):
        r = np.diff(np.log(bars["close"].to_numpy()))
        return float(np.corrcoef(r[:-1], r[1:])[0, 1])

    flat, _ = data.synthetic_daily(n_days=2000, autocorr=0.0, seed=176)
    trend, _ = data.synthetic_daily(n_days=2000, autocorr=0.20, seed=176)
    rev, _ = data.synthetic_daily(n_days=2000, autocorr=-0.20, seed=176)

    assert abs(lag1_ac(flat)) < 0.08      # near zero
    assert lag1_ac(trend) > 0.10          # positive persistence
    assert lag1_ac(rev) < -0.10           # negative (mean-reverting)


def test_fetch_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_gspc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin):
    bars, _ = coin
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=1000, autocorr=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)
