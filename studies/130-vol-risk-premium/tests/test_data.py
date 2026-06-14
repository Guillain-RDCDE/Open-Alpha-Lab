"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_risk_premium import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert set(["VIX", "RV", "VRP", "SPY_close", "SPY_ret"]).issubset(df.columns)


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, vrp_signal=0.01, seed=10)
    b, _ = data.synthetic_daily(n_days=500, vrp_signal=0.01, seed=10)
    assert np.allclose(a["VIX"].to_numpy(), b["VIX"].to_numpy())
    c, _ = data.synthetic_daily(n_days=500, vrp_signal=0.01, seed=11)
    assert not np.allclose(a["VIX"].to_numpy(), c["VIX"].to_numpy())


def test_synthetic_vix_is_positive(null_tape):
    df, _ = null_tape
    assert (df["VIX"] > 0).all()


def test_synthetic_spy_close_is_positive(null_tape):
    df, _ = null_tape
    assert (df["SPY_close"] > 0).all()


def test_synthetic_vrp_is_nondegenerate(null_tape):
    """The VRP (IV - RV) should be non-constant and have meaningful dispersion."""
    df, _ = null_tape
    vrp = df["VRP"].dropna()
    assert len(vrp) > 100
    assert vrp.std() > 0.5  # at least 0.5 pp of dispersion


def test_synthetic_vrp_positive_on_average(null_tape):
    """The synthetic tape should exhibit IV > RV on average (by construction)."""
    df, _ = null_tape
    assert df["VRP"].dropna().mean() > 0


def test_synthetic_rv_window(null_tape):
    """RV is NaN for the first RV_WINDOW rows, then defined."""
    df, _ = null_tape
    rv = df["RV"]
    assert rv.iloc[:data.RV_WINDOW].isna().all()
    assert rv.iloc[data.RV_WINDOW:].notna().all()


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_days=2000, vrp_signal=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))
