"""Tests for the data layer — pop table, synthetic panels, and cache guard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipo_pop import data  # noqa: E402

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
HAVE_CACHE = any(
    f.startswith("prices_219_") and f.endswith(".parquet")
    for f in (os.listdir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else [])
)


# ---------------------------------------------------------------------------
# Hardcoded pop table
# ---------------------------------------------------------------------------

def test_pop_table_not_empty():
    df = data.build_pop_table()
    assert not df.empty
    assert len(df) >= 30


def test_pop_table_columns():
    df = data.build_pop_table()
    for col in ("ticker", "ipo_date", "offer_price", "first_close", "pop"):
        assert col in df.columns


def test_pop_values_are_finite():
    df = data.build_pop_table()
    assert np.all(np.isfinite(df["pop"].values))


def test_pop_values_bounded():
    """Pops should be in (-1, +20) range — implausible values signal a data error."""
    df = data.build_pop_table()
    assert (df["pop"] > -0.5).all(), "Some pops < -50%, check table"
    assert (df["pop"] < 20.0).all(), "Some pops > 2000%, check table"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------

def test_synthetic_ipos_shape_and_columns(null_ipos):
    ipos, truth = null_ipos
    assert not ipos.empty
    for col in ("ticker", "ipo_date", "offer_price", "first_close", "pop",
                "ret_6m", "ret_12m", "ret_36m"):
        assert col in ipos.columns


def test_synthetic_ipos_is_deterministic():
    a, _ = data.synthetic_ipos(n_ipos=20, drift_bps=0.0, seed=7)
    b, _ = data.synthetic_ipos(n_ipos=20, drift_bps=0.0, seed=7)
    assert list(a["ret_12m"].dropna().values) == list(b["ret_12m"].dropna().values)


def test_synthetic_ipos_different_seed():
    a, _ = data.synthetic_ipos(n_ipos=20, drift_bps=0.0, seed=7)
    c, _ = data.synthetic_ipos(n_ipos=20, drift_bps=0.0, seed=99)
    assert a["ret_12m"].dropna().mean() != c["ret_12m"].dropna().mean()


def test_drift_knob_shifts_mean(null_ipos, planted_ipos):
    null_ev, _ = null_ipos
    plant_ev, _ = planted_ipos
    # Planted negative drift should produce lower 12m mean
    assert plant_ev["ret_12m"].dropna().mean() < null_ev["ret_12m"].dropna().mean()


def test_pop_knob_shifts_mean_pop():
    low, _ = data.synthetic_ipos(n_ipos=40, pop_bps=500.0, seed=1)
    high, _ = data.synthetic_ipos(n_ipos=40, pop_bps=5000.0, seed=1)
    assert high["pop"].mean() > low["pop"].mean()


def test_fingerprint_stable(null_ipos):
    ipos, _ = null_ipos
    fp1 = data.fingerprint(ipos)
    fp2 = data.fingerprint(ipos)
    assert fp1 == fp2


def test_fingerprint_sensitive(null_ipos):
    ipos, _ = null_ipos
    other, _ = data.synthetic_ipos(n_ipos=30, drift_bps=0.0, seed=999)
    assert data.fingerprint(ipos) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Cache-only guard
# ---------------------------------------------------------------------------

def test_fetch_ipo_panel_cache_only_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_ipo_panel(fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not HAVE_CACHE, reason="real-tape cache absent offline/CI")
def test_fetch_ipo_panel_from_cache():
    df = data.fetch_ipo_panel(fetch=False, cache_dir=CACHE_DIR)
    assert not df.empty
    assert "pop" in df.columns
    assert "ret_12m" in df.columns
