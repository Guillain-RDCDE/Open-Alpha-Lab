"""Tests for the data layer (hardcoded SI snapshot, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from short_interest import data  # noqa: E402

CACHE_PATH = data.FORWARD_CACHE


# ---------------------------------------------------------------------------
# Hardcoded short-interest snapshot
# ---------------------------------------------------------------------------
def test_si_table_shape_and_columns():
    si = data.short_interest_table()
    assert si.shape[0] >= 40, "Expected a reasonably sized basket"
    assert list(si.columns) == ["short_pct_float", "days_to_cover"]
    assert si.index.is_unique, "Tickers must be unique"


def test_si_table_values_sane():
    si = data.short_interest_table()
    assert (si["short_pct_float"] >= 0).all()
    assert (si["short_pct_float"] <= 60).all(), "Short % of float should be < 60%"
    assert (si["days_to_cover"] > 0).all()


def test_si_table_has_dispersion():
    """The snapshot must contain both heavily and lightly shorted names."""
    si = data.short_interest_table()
    assert si["short_pct_float"].max() >= 15, "Need crowded shorts in the basket"
    assert si["short_pct_float"].min() <= 1.5, "Need lightly shorted blue chips"


def test_si_table_deterministic():
    a = data.short_interest_table()
    b = data.short_interest_table()
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_panel(n_firms=120, si_beta=-0.03, seed=1)
    assert df.shape == (120, 2)
    assert list(df.columns) == ["short_pct_float", "fwd_ret"]
    assert truth["n_firms"] == 120
    assert truth["si_beta"] == -0.03


def test_synthetic_panel_si_nonnegative():
    df, _ = data.synthetic_panel(n_firms=200, si_beta=0.0, seed=2)
    assert (df["short_pct_float"] >= 0).all()


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(n_firms=100, si_beta=-0.04, seed=42)
    b, _ = data.synthetic_panel(n_firms=100, si_beta=-0.04, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_null_finite():
    df, truth = data.synthetic_panel(n_firms=150, si_beta=0.0, seed=10)
    assert truth["si_beta"] == 0.0
    assert np.isfinite(df["fwd_ret"]).all()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    si = data.short_interest_table()
    fp = data.fingerprint(si)
    assert isinstance(fp, str)
    assert len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(n_firms=80, si_beta=-0.02, seed=7)
    b, _ = data.synthetic_panel(n_firms=80, si_beta=-0.06, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard: offline by default
# ---------------------------------------------------------------------------
def test_fetch_requires_cache_when_offline():
    """With fetch=False and no cache, fetch_forward_returns must raise (no network)."""
    if os.path.exists(CACHE_PATH):
        pytest.skip("cache present; offline-guard test not applicable")
    with pytest.raises(FileNotFoundError):
        data.fetch_forward_returns(fetch=False, cache_path=CACHE_PATH)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="si_forward.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_forward_returns_shape():
    fwd = data.fetch_forward_returns(fetch=False, cache_path=CACHE_PATH)
    assert fwd.shape[0] > 20, "Expected forward returns for >20 tickers"
    assert any(c.startswith("fwd_") for c in fwd.columns)


@requires_cache
def test_real_panel_join():
    panel = data.real_panel(horizon_col="fwd_3m", cache_path=CACHE_PATH)
    assert "short_pct_float" in panel.columns
    assert "fwd_ret" in panel.columns
    assert len(panel) > 20
