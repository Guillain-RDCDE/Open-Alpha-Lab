"""Synthetic tape is well-formed and deterministic; Shiller loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sad_effect import data  # noqa: E402


def test_synthetic_shape(null_tape):
    ret, truth = null_tape
    assert len(ret) == truth["n_years"] * 12
    assert ret.index.freq is not None or len(ret) > 0
    assert ret.isna().sum() == 0


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_years=30, sad_premium_bp=0.0, seed=5)
    b, _ = data.synthetic_monthly(n_years=30, sad_premium_bp=0.0, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_monthly(n_years=30, sad_premium_bp=0.0, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_null_tape_has_no_planted_sad(null_tape):
    ret, truth = null_tape
    assert truth["sad_premium_bp"] == 0.0
    assert truth["has_sad"] is False


def test_sad_tape_plants_the_premium(with_sad):
    ret, truth = with_sad
    assert truth["has_sad"] is True
    assert truth["sad_premium_bp"] == 40.0


def test_synthetic_month_coverage(null_tape):
    ret, _ = null_tape
    months = ret.index.month
    # All 12 months must appear in an 80-year series
    assert set(months.unique()) == set(range(1, 13))


def test_daylight_proxy_shape_and_range(null_tape):
    ret, _ = null_tape
    dl = data.daylight_proxy(ret.index)
    assert len(dl) == len(ret)
    # First value is NaN (no prior month)
    assert np.isnan(dl.iloc[0])
    # After that all should be finite
    assert dl.iloc[1:].isna().sum() == 0


def test_daylight_proxy_seasonality():
    """Day-length increases in spring and decreases in autumn for Northern Hemisphere."""
    import pandas as pd

    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    dl = data.daylight_proxy(idx)
    # January→February: days lengthen (positive change)
    assert dl.iloc[1] > 0  # Feb vs Jan
    # June→July: days shorten (negative change, past summer solstice)
    assert dl.iloc[6] < 0  # Jul vs Jun


def test_fingerprint_stable_and_content_sensitive(null_tape):
    ret, _ = null_tape
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _ = data.synthetic_monthly(n_years=80, sad_premium_bp=0.0, seed=99)
    assert data.fingerprint(ret) != data.fingerprint(other)


def test_shiller_raises_without_cache(tmp_path):
    """Cache-only loader raises FileNotFoundError on a missing cache directory."""
    with pytest.raises(FileNotFoundError):
        data.fetch_shiller(cache_dir=str(tmp_path))
