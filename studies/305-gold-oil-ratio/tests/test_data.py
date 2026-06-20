"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_oil_ratio import data  # noqa: E402

CACHE = data.DEFAULT_CACHE


def test_synthetic_shape_and_columns(null):
    daily, truth = null
    assert len(daily) == truth["n_days"]
    assert list(daily.columns) == ["gold", "oil", "equity", "rf"]
    assert (daily["gold"] > 0).all()
    assert (daily["oil"] > 0).all()
    assert (daily["equity"] > 0).all()
    # rf is a constant positive daily rate.
    assert (daily["rf"] > 0).all()
    assert np.isclose(daily["rf"].std(), 0.0)


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, pred_r=-0.01, seed=7)
    b, _ = data.synthetic_daily(n_days=500, pred_r=-0.01, seed=7)
    assert np.allclose(a["equity"].to_numpy(), b["equity"].to_numpy())
    c, _ = data.synthetic_daily(n_days=500, pred_r=-0.01, seed=8)
    assert not np.allclose(a["equity"].to_numpy(), c["equity"].to_numpy())


def test_null_has_no_ratio_equity_predictability(null):
    """With pred_r=0 the next equity return is uncorrelated with the lagged log ratio."""
    daily, _ = null
    log_ratio = np.log(daily["gold"] / daily["oil"]).to_numpy()
    eq_r = np.diff(np.log(daily["equity"].to_numpy()))
    # corr(lagged ratio level change, next equity return) should be ~0.
    drat = np.diff(log_ratio)
    c = np.corrcoef(drat[:-1], eq_r[1:])[0, 1]
    assert abs(c) < 0.06


def test_planted_creates_predictability(planted):
    """pred_r<0: a high standardised ratio precedes a lower next equity return (neg corr)."""
    daily, _ = planted
    lr = np.log(daily["gold"] / daily["oil"])
    z = ((lr - lr.rolling(60).mean()) / lr.rolling(60).std(ddof=1)).to_numpy()
    eq_r = daily["equity"].pct_change().to_numpy()
    mask = np.isfinite(z[:-1]) & np.isfinite(eq_r[1:])
    c = np.corrcoef(z[:-1][mask], eq_r[1:][mask])[0, 1]
    assert c < -0.05  # high ratio → lower forward return


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_load_real_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


def test_have_real_cache_false_on_empty_dir(tmp_path):
    assert data.have_real_cache(cache_dir=str(tmp_path)) is False


def test_fingerprint_is_stable_and_content_sensitive(null):
    daily, _ = null
    assert data.fingerprint(daily) == data.fingerprint(daily)
    other, _ = data.synthetic_daily(n_days=4000, pred_r=0.0, seed=99)
    assert data.fingerprint(daily) != data.fingerprint(other)


@pytest.mark.skipif(not data.have_real_cache(CACHE), reason="offline CI: no real cache")
def test_load_real_assembles_when_cache_present():
    df = data.load_real(fetch=False)
    assert list(df.columns) == ["gold", "oil", "equity", "rf"]
    assert len(df) > 1000
    assert df.index.is_monotonic_increasing
