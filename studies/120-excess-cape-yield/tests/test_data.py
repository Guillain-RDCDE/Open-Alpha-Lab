"""Tests for the data layer — synthetic generator and Shiller parquet loader."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from excess_cape_yield import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(signal_world):
    frame, truth = signal_world
    expected_cols = {"cape", "earnings_yield", "real_bond_yield", "ecy", "fwd10_excess", "fwd1_excess"}
    assert expected_cols.issubset(set(frame.columns))
    assert len(frame) == truth["n_years"] * 12


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(n_years=50, predicts=1.0, seed=42)
    b, _ = data.synthetic_world(n_years=50, predicts=1.0, seed=42)
    assert np.allclose(a["ecy"].to_numpy(), b["ecy"].to_numpy())
    c, _ = data.synthetic_world(n_years=50, predicts=1.0, seed=43)
    assert not np.allclose(a["ecy"].to_numpy(), c["ecy"].to_numpy())


def test_synthetic_null_has_no_ecy_structure(null_world):
    """In the null world, ECY and fwd10_excess should be near-uncorrelated."""
    frame, _ = null_world
    sub = frame[["ecy", "fwd10_excess"]].dropna().iloc[::120]
    if len(sub) < 3:
        pytest.skip("too few non-overlapping windows for correlation")
    corr = float(np.corrcoef(sub["ecy"].values, sub["fwd10_excess"].values)[0, 1])
    assert abs(corr) < 0.7, f"Null world should have low ECY-fwd correlation, got {corr:.2f}"


def test_synthetic_signal_has_ecy_structure(signal_world):
    """In the signal world ECY should strongly predict fwd10_excess."""
    frame, _ = signal_world
    sub = frame[["ecy", "fwd10_excess"]].dropna().iloc[::120]
    if len(sub) < 3:
        pytest.skip("too few non-overlapping windows for correlation")
    corr = float(np.corrcoef(sub["ecy"].values, sub["fwd10_excess"].values)[0, 1])
    assert corr > 0.5, f"Signal world should have high ECY-fwd correlation, got {corr:.2f}"


def test_synthetic_ecy_equals_ey_minus_bond(signal_world):
    """ECY = earnings_yield − real_bond_yield (the definitional identity)."""
    frame, _ = signal_world
    computed = frame["earnings_yield"] - frame["real_bond_yield"]
    assert np.allclose(computed.to_numpy(), frame["ecy"].to_numpy(), atol=1e-12)


def test_synthetic_cape_positive(signal_world):
    frame, _ = signal_world
    assert (frame["cape"] > 0).all()


def test_fingerprint_stable_and_content_sensitive(signal_world):
    frame, _ = signal_world
    fp1 = data.fingerprint(frame)
    fp2 = data.fingerprint(frame)
    assert fp1 == fp2
    other, _ = data.synthetic_world(n_years=130, predicts=1.0, seed=99)
    assert fp1 != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real data loader (skipped if the parquet is absent)
# ---------------------------------------------------------------------------
SHILLER_PRESENT = os.path.exists(data.SHILLER_FILE)


@pytest.mark.skipif(not SHILLER_PRESENT, reason="Shiller parquet not staged")
def test_shiller_columns():
    df = data.load_shiller()
    for col in ("ecy", "earnings_yield", "real_bond_yield", "fwd10_excess", "fwd1_excess"):
        assert col in df.columns, f"Missing column {col}"


@pytest.mark.skipif(not SHILLER_PRESENT, reason="Shiller parquet not staged")
def test_shiller_ecy_identity():
    """ECY = earnings_yield - real_bond_yield must hold exactly."""
    df = data.load_shiller()
    diff = (df["ecy"] - (df["earnings_yield"] - df["real_bond_yield"])).abs()
    assert diff.max() < 1e-10, "ECY identity broken"


@pytest.mark.skipif(not SHILLER_PRESENT, reason="Shiller parquet not staged")
def test_shiller_no_zero_pe10():
    """PE10 = 0 rows must have been dropped before any column is computed."""
    df = data.load_shiller()
    assert (df["PE10"] > 0).all()


@pytest.mark.skipif(not SHILLER_PRESENT, reason="Shiller parquet not staged")
def test_shiller_long_enough():
    df = data.load_shiller()
    assert len(df) > 1000, "Expected > 1000 monthly rows"
    assert df.index[0].year <= 1895
