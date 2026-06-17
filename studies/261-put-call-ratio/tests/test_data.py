"""Tests for the data layer (hardcoded put/call table, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from put_call_ratio import data  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Hardcoded put/call table
# ---------------------------------------------------------------------------
def test_pcr_series_month_end_sorted():
    s = data.pcr_series()
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.is_monotonic_increasing
    for ts in s.index[:6]:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} not month-end"


def test_pcr_values_plausible():
    s = data.pcr_series()
    # Put/call ratios are positive and live in a sane band.
    assert (s > 0.4).all() and (s < 2.0).all()
    # Long-run mean near ~0.92 (folklore baseline).
    assert 0.85 < s.mean() < 1.0


def test_pcr_crisis_spikes():
    """Known panic months should print high readings."""
    s = data.pcr_series()
    for ym in ("2008-10", "2018-12", "2020-03"):
        ts = pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0)
        assert s.loc[ts] >= 1.15, f"{ym} should be an extreme-fear month"


def test_pcr_length():
    s = data.pcr_series()
    # 2003-01 .. 2025-12 inclusive = 23 years * 12 = 276 months.
    assert len(s) == 276


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_panel(n_months=120, seed=1)
    assert df.shape == (120, 2)
    assert set(df.columns) == {"pcr", "fwd_ret"}
    assert truth["n_months"] == 120


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(contrarian_beta=0.15, seed=42)
    b, _ = data.synthetic_panel(contrarian_beta=0.15, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_month_end_index():
    df, _ = data.synthetic_panel(n_months=60, seed=5)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_synthetic_null_no_beta():
    df, truth = data.synthetic_panel(contrarian_beta=0.0, seed=10)
    assert truth["contrarian_beta"] == 0.0
    assert np.isfinite(df["fwd_ret"]).all()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_panel(n_months=60, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(contrarian_beta=0.0, seed=7)
    b, _ = data.synthetic_panel(contrarian_beta=0.30, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
def test_fetch_offline_raises_when_absent(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_monthly(fetch=False, cache_path=missing)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    df = data.build_real_panel(fetch=False)
    assert df.shape[0] > 100
    assert set(df.columns) == {"pcr", "fwd_ret"}


@requires_cache
def test_real_panel_aligned_month_end():
    df = data.build_real_panel(fetch=False)
    for ts in df.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)
