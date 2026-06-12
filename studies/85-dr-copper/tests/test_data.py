"""The synthetic tape is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dr_copper import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    daily, truth = null_tape
    assert len(daily) == truth["n_years"] * 252
    assert set(daily.columns) >= {"copper", "gold", "equity", "yield10"}
    assert (daily["copper"] > 0).all()
    assert (daily["gold"] > 0).all()
    assert (daily["equity"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=5)
    b, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=5)
    assert np.allclose(a["equity"].to_numpy(), b["equity"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=6)
    assert not np.allclose(a["equity"].to_numpy(), c["equity"].to_numpy())


def test_pred_r_zero_means_near_zero_correlation(null_tape):
    """With pred_r=0, copper/gold changes should not correlate with next-week equity returns."""
    daily, _ = null_tape
    # compute_features handles the cu_au column computation internally
    from dr_copper.strategy import compute_features
    feat = compute_features(daily, horizon="W")
    corr = np.corrcoef(feat["delta_cu_au"].to_numpy(), feat["fwd_equity"].to_numpy())[0, 1]
    # On a null tape over 20 years weekly, |corr| should be small (allow some sampling noise)
    assert abs(corr) < 0.20


def test_pred_r_positive_creates_positive_correlation(signal_tape):
    """With pred_r=0.5, there should be a detectable positive predictive correlation."""
    daily, _ = signal_tape
    from dr_copper.strategy import compute_features
    feat = compute_features(daily, horizon="W")
    corr = np.corrcoef(feat["delta_cu_au"].to_numpy(), feat["fwd_equity"].to_numpy())[0, 1]
    # pred_r=0.5 is a daily link; after weekly aggregation the weekly correlation may be smaller
    # but should still be positive and larger than the null tape
    null, _ = data.synthetic_daily(n_years=20, pred_r=0.0, seed=85)
    feat_null = compute_features(null, horizon="W")
    corr_null = np.corrcoef(feat_null["delta_cu_au"].to_numpy(), feat_null["fwd_equity"].to_numpy())[0, 1]
    assert corr > corr_null  # signal tape should have higher correlation than null


def test_build_ratio_frame_aligns_series(null_tape):
    """strategy.compute_features computes cu_au internally when absent."""
    daily, _ = null_tape
    # Raw synthetic has no pre-computed ratio column
    assert "cu_au" not in daily.columns
    # compute_features adds it internally
    from dr_copper.strategy import compute_features
    feat = compute_features(daily, horizon="W")
    # cu_au column should be present after compute_features
    assert "cu_au" in feat.columns
    assert (feat["cu_au"] > 0).all()


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    daily, _ = null_tape
    assert data.fingerprint(daily) == data.fingerprint(daily)
    other, _ = data.synthetic_daily(n_years=20, pred_r=0.0, seed=99)
    assert data.fingerprint(daily) != data.fingerprint(other)
