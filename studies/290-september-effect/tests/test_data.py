"""Tests for the data layer of Study 290 (September-Effect).

All tests are offline and deterministic -- they use the synthetic generator
only and confirm the real-data loader refuses to touch the network.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from september_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic generator -- shape and columns
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"] * 12
    assert set(df.columns) >= {"date", "year", "month", "ret", "is_sep"}


def test_synthetic_months_cycle(synthetic_null):
    df, _ = synthetic_null
    assert set(df["month"].unique()) == set(range(1, 13))
    # 76 years -> 76 Septembers
    assert int((df["month"] == 9).sum()) == 76
    assert (df["is_sep"] == (df["month"] == 9)).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_years=30, sep_bps=0.0, seed=42)
    b, _ = data.synthetic_monthly(n_years=30, sep_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_monthly(n_years=30, sep_bps=0.0, seed=1)
    b, _ = data.synthetic_monthly(n_years=30, sep_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


# ---------------------------------------------------------------------------
# The null vs the planted drag -- the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_september_drag():
    """On the null tape with large n, September mean ~ other-month mean."""
    df, _ = data.synthetic_monthly(n_years=2000, sep_bps=0.0, seed=290)
    sep = df.loc[df["month"] == 9, "ret"].mean()
    oth = df.loc[df["month"] != 9, "ret"].mean()
    assert abs(sep - oth) < 0.003  # < 30 bps with n=2000 Septembers


def test_planted_drag_creates_gap():
    """A negative planted drag makes September meaningfully lower."""
    df, _ = data.synthetic_monthly(n_years=500, sep_bps=-1000.0, seed=290)
    sep = df.loc[df["month"] == 9, "ret"].mean()
    oth = df.loc[df["month"] != 9, "ret"].mean()
    assert sep < oth - 0.05  # planted -1000 bps/mo dominates the noise


def test_synthetic_zero_drag_is_special_case():
    """sep_bps=0 is exactly the same draw as the null fixture (determinism)."""
    a, _ = data.synthetic_monthly(n_years=10, sep_bps=0.0, seed=290)
    b, _ = data.synthetic_monthly(n_years=10, sep_bps=0.0, seed=290)
    assert np.array_equal(a["ret"].to_numpy(), b["ret"].to_numpy())


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_monthly(n_years=20, sep_bps=0.0, seed=10)
    df2, _ = data.synthetic_monthly(n_years=20, sep_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real loader stays offline by default
# ---------------------------------------------------------------------------
def test_fetch_raises_without_cache(tmp_path):
    """With no Shiller cache and fetch=False, fetch_sp500_monthly raises."""
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_monthly(cache_dir=str(tmp_path))


def test_fetch_default_is_offline(monkeypatch):
    """fetch=False must never import yfinance / touch the network."""
    def _boom(*a, **k):  # pragma: no cover - should not be hit
        raise AssertionError("network path taken with fetch=False")

    monkeypatch.setattr(data, "_fetch_gspc_live", _boom)
    # Missing cache dir -> FileNotFoundError (not the network boom).
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_monthly(cache_dir="/nonexistent_cache_dir_xyz")
