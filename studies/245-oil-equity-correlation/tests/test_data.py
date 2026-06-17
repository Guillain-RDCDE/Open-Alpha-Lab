"""The synthetic tape is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oil_equity_correlation import data  # noqa: E402

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)


def test_synthetic_shape_and_columns(null_tape):
    daily, truth = null_tape
    assert len(daily) == truth["n_years"] * 252
    assert set(daily.columns) >= {"oil", "equity"}
    assert (daily["oil"] > 0).all()
    assert (daily["equity"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=5)
    b, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=5)
    assert np.allclose(a["equity"].to_numpy(), b["equity"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, pred_r=0.1, seed=6)
    assert not np.allclose(a["equity"].to_numpy(), c["equity"].to_numpy())


def test_pred_r_zero_means_near_zero_correlation(null_tape):
    """With pred_r=0, oil returns should not correlate with next-week equity returns."""
    daily, _ = null_tape
    from oil_equity_correlation.strategy import compute_features
    feat = compute_features(daily, horizon="W")
    corr = np.corrcoef(feat["ret_oil"].to_numpy(), feat["fwd_equity"].to_numpy())[0, 1]
    # On a null tape over 20 years weekly, |corr| should be small (allow some sampling noise)
    assert abs(corr) < 0.20


def test_pred_r_positive_creates_positive_correlation(signal_tape):
    """With pred_r=0.5, there should be a detectable positive predictive correlation."""
    daily, _ = signal_tape
    from oil_equity_correlation.strategy import compute_features
    feat = compute_features(daily, horizon="W")
    corr = np.corrcoef(feat["ret_oil"].to_numpy(), feat["fwd_equity"].to_numpy())[0, 1]
    null, _ = data.synthetic_daily(n_years=20, pred_r=0.0, seed=245)
    feat_null = compute_features(null, horizon="W")
    corr_null = np.corrcoef(feat_null["ret_oil"].to_numpy(), feat_null["fwd_equity"].to_numpy())[0, 1]
    assert corr > corr_null  # signal tape should have higher correlation than null


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    daily, _ = null_tape
    assert data.fingerprint(daily) == data.fingerprint(daily)
    other, _ = data.synthetic_daily(n_years=20, pred_r=0.0, seed=99)
    assert data.fingerprint(daily) != data.fingerprint(other)


@pytest.mark.skipif(
    not all(
        os.path.exists(os.path.join(CACHE_DIR, f"daily_{t.replace('=','').replace('^','').replace('/','')}.parquet"))
        for t in data.TICKERS
    ),
    reason="real-tape cache absent offline/CI",
)
def test_real_tape_loads_and_aligns():
    panel = data.load_panel(fetch=False, cache_dir=CACHE_DIR)
    daily = data.build_panel_frame(panel)
    assert len(daily) > 1000
    assert set(daily.columns) >= {"oil", "equity"}
    assert (daily["oil"] > 0).all()
    assert (daily["equity"] > 0).all()
