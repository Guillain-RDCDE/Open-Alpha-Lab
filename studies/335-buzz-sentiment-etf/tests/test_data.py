"""The synthetic pair is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buzz_sentiment_etf import data  # noqa: E402

CACHE = data.DEFAULT_CACHE


def test_synthetic_shape_and_levels(null_pair):
    levels, truth = null_pair
    assert len(levels) == truth["n_days"]
    assert list(levels.columns) == ["buzz", "spy"]
    assert (levels > 0).all().all()
    # both legs rebased to 100 at the first bar
    assert abs(levels["buzz"].iloc[0] - 100.0) < 1e-6
    assert abs(levels["spy"].iloc[0] - 100.0) < 1e-6


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_pair(n_days=400, alpha_bps=2.0, seed=7)
    b, _ = data.synthetic_pair(n_days=400, alpha_bps=2.0, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_pair(n_days=400, alpha_bps=2.0, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_alpha_knob_lifts_buzz_relative_to_spy():
    """Planting alpha must raise BUZZ's terminal value relative to the null tape."""
    null, _ = data.synthetic_pair(n_days=3000, alpha_bps=0.0, seed=1)
    pos, _ = data.synthetic_pair(n_days=3000, alpha_bps=6.0, seed=1)
    # SPY leg is identical (same seed, same draws); BUZZ leg should be higher with alpha.
    assert np.allclose(null["spy"].to_numpy(), pos["spy"].to_numpy())
    assert pos["buzz"].iloc[-1] > null["buzz"].iloc[-1]


def test_truth_reports_annual_alpha(alpha_pair):
    _, truth = alpha_pair
    # 4 bps/day * 252 / 100 ≈ 10.08 %/yr
    assert abs(truth["alpha_bps_ann"] - 4.0 * 252 / 100.0) < 1e-9


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_pair):
    levels, _ = null_pair
    assert data.fingerprint(levels) == data.fingerprint(levels)
    other, _ = data.synthetic_pair(n_days=3000, alpha_bps=0.0, seed=99)
    assert data.fingerprint(levels) != data.fingerprint(other)


@pytest.mark.skipif(
    not (os.path.exists(data._cache_path("BUZZ", CACHE))
         and os.path.exists(data._cache_path("SPY", CACHE))),
    reason="offline CI — no real BUZZ/SPY cache",
)
def test_load_real_from_cache_shape():
    df = data.load_real(fetch=False)
    assert list(df.columns) == ["buzz", "spy"]
    assert len(df) > 100
    assert abs(df.iloc[0]["spy"] - 100.0) < 1e-6
