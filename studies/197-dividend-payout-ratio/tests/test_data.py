"""Tests for the data layer — Study 197 (Dividend-Payout-Ratio).

Synthetic-tape tests (no cache) cover all data-engine logic; real-data tests are
guarded with ``@requires_cache`` and skip on CI runners.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_payout_ratio import data  # noqa: E402

# Cache guard: skip real-data tests when the Shiller parquet is absent (CI).
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.SHILLER_CACHE),
    reason="Shiller cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"] * 12
    assert set(df.columns) == {"payout", "fwd_outcome"}
    assert df["payout"].notna().all()
    assert df["fwd_outcome"].notna().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_years=50, payout_signal=0.05, seed=7)
    b, _ = data.synthetic_monthly(n_years=50, payout_signal=0.05, seed=7)
    assert np.allclose(a["payout"].to_numpy(), b["payout"].to_numpy())
    c, _ = data.synthetic_monthly(n_years=50, payout_signal=0.05, seed=8)
    assert not np.allclose(a["payout"].to_numpy(), c["payout"].to_numpy())


def test_synthetic_payout_positive_and_bounded(null_tape):
    df, _ = null_tape
    assert (df["payout"] > 0).all()
    assert (df["payout"] <= 2.0).all()


def test_synthetic_null_has_near_zero_slope(null_tape):
    """On the null tape (no signal), the payout slope on the outcome is near zero."""
    df, _ = null_tape
    from dividend_payout_ratio import strategy as st
    reg = st.regress_synthetic(df)
    # With no signal planted, |slope| should be small; this is not guaranteed to
    # be < 2σ in every seed but the null tape is large enough that the slope is
    # well inside noise.
    assert abs(reg["slope"]) < 0.08, f"null slope unexpectedly large: {reg['slope']}"


def test_synthetic_signal_tape_has_positive_slope(signal_tape):
    """With a positive payout_signal, the regression recovers a positive slope."""
    df, truth = signal_tape
    from dividend_payout_ratio import strategy as st
    reg = st.regress_synthetic(df)
    assert reg["slope"] > 0.0, f"expected positive slope, got {reg['slope']}"


def test_synthetic_signal_stronger_than_null():
    """The planted signal must produce a steeper slope than the null."""
    from dividend_payout_ratio import strategy as st
    df_null, _ = data.synthetic_monthly(n_years=80, payout_signal=0.0, seed=42)
    df_sig, _ = data.synthetic_monthly(n_years=80, payout_signal=0.15, seed=42)
    r_null = st.regress_synthetic(df_null)
    r_sig = st.regress_synthetic(df_sig)
    assert r_sig["slope"] > r_null["slope"]


def test_fingerprint_stable_and_content_sensitive():
    df, _ = data.synthetic_monthly(n_years=30, payout_signal=0.0, seed=1)
    # Make payout a column present in the frame
    df = df.rename(columns={"payout": "Payout"})
    fp1 = data.fingerprint(df)
    fp2 = data.fingerprint(df)
    assert fp1 == fp2  # stable
    df2, _ = data.synthetic_monthly(n_years=30, payout_signal=0.0, seed=2)
    df2 = df2.rename(columns={"payout": "Payout"})
    assert data.fingerprint(df) != data.fingerprint(df2)  # content-sensitive


def test_load_shiller_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_shiller(cache_path=str(tmp_path / "nonexistent.parquet"))


# ---------------------------------------------------------------------------
# Real data tests — skipped when cache is absent
# ---------------------------------------------------------------------------
@requires_cache
def test_shiller_columns_and_non_null():
    df = data.load_shiller()
    for col in ["Payout", "fwd_10y_eg", "fwd_10y_price"]:
        assert col in df.columns, f"missing column {col}"
    assert df["Payout"].notna().all()
    assert df["fwd_10y_eg"].notna().all()
    assert df["fwd_10y_price"].notna().all()


@requires_cache
def test_shiller_payout_range():
    df = data.load_shiller()
    # Payout should be between 0 and 3 for economically valid periods
    assert (df["Payout"] > 0).all()
    assert (df["Payout"] < 3.0).all()


@requires_cache
def test_shiller_length_and_date_range():
    df = data.load_shiller()
    # Should span from ~1882 (12m rolling needs 12 obs) to ~2013 (last with 10y ahead)
    assert df.index.min().year <= 1885
    assert df.index.max().year >= 2010
    assert len(df) > 1500


@requires_cache
def test_fingerprint_on_real_data():
    df = data.load_shiller()
    fp = data.fingerprint(df)
    assert isinstance(fp, str)
    assert len(fp) == 12
    # Should be identical on repeat call
    assert fp == data.fingerprint(df)
