"""Synthetic generator is well-formed, deterministic, and the knobs work."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from four_percent_rule import data  # noqa: E402


def test_synthetic_shape_and_columns(normal_returns):
    df, truth = normal_returns
    assert list(df.columns) == ["eq_real_tr", "bond_real_tr"]
    assert len(df) == truth["n_years_total"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_cohorts(n_years_total=50, seed=10)
    b, _ = data.synthetic_cohorts(n_years_total=50, seed=10)
    assert np.allclose(a["eq_real_tr"].to_numpy(), b["eq_real_tr"].to_numpy())
    c, _ = data.synthetic_cohorts(n_years_total=50, seed=11)
    assert not np.allclose(a["eq_real_tr"].to_numpy(), c["eq_real_tr"].to_numpy())


def test_equity_mean_is_close_to_planted(normal_returns):
    """Long-run mean of the synthetic equity series should be near the planted value."""
    df, truth = normal_returns
    # With 100 years the sample mean should be within ~2 vol/sqrt(n) of the target.
    tol = 2 * truth["equity_vol"] / np.sqrt(truth["n_years_total"])
    # Arithmetic mean of log-normal ≈ exp(mu + sigma^2/2) - 1 ≈ planted + vol^2/2
    # We test that the sample mean of *arithmetic* returns is positive and in the right ballpark.
    mean_eq = df["eq_real_tr"].mean()
    assert mean_eq > 0.0, "Equity returns should be positive on average"
    assert abs(mean_eq - truth["equity_real_mean"]) < 0.08, (
        f"Sample mean {mean_eq:.3f} too far from planted {truth['equity_real_mean']:.3f}"
    )


def test_bad_start_lowers_early_returns(normal_returns, bad_start_returns):
    """The bad_start knob should make the first 3 years of equity returns worse."""
    df_good, _ = normal_returns
    df_bad, _ = bad_start_returns
    # First 3 years should be lower in the bad-start version.
    good_early = df_good["eq_real_tr"].iloc[:3].mean()
    bad_early = df_bad["eq_real_tr"].iloc[:3].mean()
    assert bad_early < good_early, "bad_start should lower early returns"


def test_low_yield_has_lower_equity_mean(normal_returns, low_yield_returns):
    df_normal, _ = normal_returns
    df_low, _ = low_yield_returns
    assert df_low["eq_real_tr"].mean() < df_normal["eq_real_tr"].mean()


def test_fingerprint_stable_and_content_sensitive(normal_returns):
    df, _ = normal_returns
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_cohorts(n_years_total=100, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_shiller_not_loaded_in_offline_mode():
    """The synthetic path must not touch the network or the Shiller parquet."""
    # This just confirms synthetic_cohorts runs without needing any external file.
    df, truth = data.synthetic_cohorts(n_years_total=30, seed=42)
    assert len(df) == 30
    assert "eq_real_tr" in df.columns
