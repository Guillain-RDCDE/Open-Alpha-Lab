"""The synthetic world is well-formed, deterministic, and carries the right structure;
the Shiller loader is idempotent and returns sensible real-data columns."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inflation_hedge import data  # noqa: E402

REQUIRED_COLS = [
    "cpi_yoy",
    "cpi_regime",
    "nom_ret_12m",
    "real_ret_12m",
    "real_tr_ret_12m",
    "fwd_real_tr_12m",
    "fwd_real_tr_60m",
]


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------
def test_synthetic_columns_present(null_world):
    df, _ = null_world
    for col in REQUIRED_COLS:
        assert col in df.columns, f"missing column: {col}"


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(n_years=50, fisher_beta=0.5, seed=7)
    b, _ = data.synthetic_world(n_years=50, fisher_beta=0.5, seed=7)
    assert np.allclose(a["cpi_yoy"].dropna().to_numpy(), b["cpi_yoy"].dropna().to_numpy())
    # Different seed → different data
    c, _ = data.synthetic_world(n_years=50, fisher_beta=0.5, seed=8)
    assert not np.allclose(a["cpi_yoy"].dropna().to_numpy(), c["cpi_yoy"].dropna().to_numpy())


def test_synthetic_regime_labels_are_binary(null_world):
    df, _ = null_world
    assert set(df["cpi_regime"].dropna().unique()) <= {"high", "low"}


def test_synthetic_no_look_ahead_in_forward_cols(null_world):
    """Forward return columns are NaN at the end of the frame (the last 12
    months have no 12m-forward horizon), confirming they really look forward."""
    df, _ = null_world
    # The last rows should be NaN for the 12m forward col
    last_12 = df["fwd_real_tr_12m"].iloc[-12:]
    assert last_12.isna().any(), (
        "Expected NaN in the last 12 rows of fwd_real_tr_12m (no future window)"
    )


def test_synthetic_fisher_beta_0_real_return_falls(null_world):
    """With beta=0 nominal returns are uncorrelated with inflation, so real
    returns should be *negatively* correlated with inflation (as in Fama-Schwert
    1977): real = nominal - inflation ≈ noise - inflation."""
    df, _ = null_world
    sub = df[["cpi_yoy", "real_tr_ret_12m"]].dropna()
    corr = np.corrcoef(sub["cpi_yoy"], sub["real_tr_ret_12m"])[0, 1]
    assert corr < 0, f"Expected negative corr (Fama-Schwert), got {corr:.3f}"


def test_synthetic_fisher_beta_1_real_return_invariant(full_hedge_world):
    """With beta=1 the nominal return fully absorbs inflation, so real returns
    should be approximately uncorrelated with inflation (near zero correlation)."""
    df, _ = full_hedge_world
    sub = df[["cpi_yoy", "real_tr_ret_12m"]].dropna()
    corr = np.corrcoef(sub["cpi_yoy"], sub["real_tr_ret_12m"])[0, 1]
    # Should be much closer to zero than in the null world (allow ±0.35)
    assert abs(corr) < 0.35, f"Expected near-zero corr with full hedge, got {corr:.3f}"


def test_fingerprint_is_stable_and_content_sensitive(null_world):
    df, _ = null_world
    fp1 = data.fingerprint(df)
    fp2 = data.fingerprint(df)
    assert fp1 == fp2
    other, _ = data.synthetic_world(n_years=100, fisher_beta=0.5, seed=99)
    assert fp1 != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real Shiller panel
# ---------------------------------------------------------------------------
def test_shiller_columns(shiller_df):
    for col in REQUIRED_COLS:
        assert col in shiller_df.columns, f"missing column: {col}"


def test_shiller_no_nan_in_core_signal_cols(shiller_df):
    """The core signal columns should have no NaN (they're used to form the
    regime label and Fisher regression)."""
    for col in ["cpi_yoy", "nom_ret_12m", "real_tr_ret_12m"]:
        assert shiller_df[col].isna().sum() == 0, f"unexpected NaN in {col}"


def test_shiller_cpi_yoy_positive_long_run(shiller_df):
    """Over the full 1872-2023 panel, mean CPI inflation is positive."""
    assert shiller_df["cpi_yoy"].mean() > 0.0


def test_shiller_has_both_regimes(shiller_df):
    regimes = set(shiller_df["cpi_regime"].unique())
    assert regimes == {"high", "low"}


def test_shiller_fingerprint_stable(shiller_df):
    fp = data.fingerprint(shiller_df)
    assert len(fp) == 12
    assert fp == data.fingerprint(shiller_df)  # idempotent
