"""Tests for the data layer (synthetic series, curated BDI, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baltic_dry import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------
def test_synthetic_shape():
    df, truth = data.synthetic_series(n_months=240, beta=0.3, seed=1)
    assert df.shape == (240, 2)
    assert list(df.columns) == ["bdi", "sp500"]
    assert truth["n_months"] == 240
    assert truth["beta"] == 0.3


def test_synthetic_prices_positive():
    df, _ = data.synthetic_series(n_months=240, beta=0.3, seed=2)
    assert (df["bdi"] > 0).all()
    assert (df["sp500"] > 0).all()


def test_synthetic_deterministic():
    a, _ = data.synthetic_series(n_months=200, beta=0.3, seed=42)
    b, _ = data.synthetic_series(n_months=200, beta=0.3, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_month_end_index():
    df, _ = data.synthetic_series(n_months=60, beta=0.0, seed=5)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} is not month-end"


def test_synthetic_null_beta_recorded():
    _, truth = data.synthetic_series(n_months=120, beta=0.0, seed=10)
    assert truth["beta"] == 0.0


# ---------------------------------------------------------------------------
# Curated BDI series
# ---------------------------------------------------------------------------
def test_curated_bdi_positive_and_monthly():
    bdi = data.curated_bdi()
    assert (bdi > 0).all()
    assert bdi.index.is_monotonic_increasing
    for ts in bdi.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_curated_bdi_hits_known_peak():
    """The curated series should peak around the May-2008 super-spike (~11k)."""
    bdi = data.curated_bdi()
    peak_date = bdi.idxmax()
    assert peak_date.year == 2008
    assert bdi.max() > 9000.0


def test_curated_bdi_launch_level():
    bdi = data.curated_bdi()
    # Index launched at 1000 in Jan 1985
    first = bdi.iloc[0]
    assert abs(first - 1000.0) < 50.0


# ---------------------------------------------------------------------------
# Real tape (offline build from Shiller + curated BDI)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.exists(data.SHILLER_CACHE),
    reason="Shiller cache absent (gitignored) — offline logic is covered by the synthetic tests",
)
def test_fetch_real_offline_builds():
    """Cache-only fetch_real builds a (bdi, sp500) frame from Shiller + curated BDI."""
    df = data.fetch_real(fetch=False)
    assert list(df.columns) == ["bdi", "sp500"]
    assert df.shape[0] > 200, "Expected a few hundred monthly rows"
    assert (df > 0).all().all()


@pytest.mark.skipif(
    not os.path.exists(data.SHILLER_CACHE),
    reason="Shiller cache absent (gitignored) — offline logic is covered by the synthetic tests",
)
def test_fetch_real_month_end_index():
    df = data.fetch_real(fetch=False)
    for ts in df.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_series(n_months=60, beta=0.3, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_series(n_months=60, beta=0.3, seed=7)
    b, _ = data.synthetic_series(n_months=60, beta=0.6, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)
