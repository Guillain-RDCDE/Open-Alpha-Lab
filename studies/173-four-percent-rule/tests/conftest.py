"""Shared fixtures — deterministic synthetic annual return series for retirement cohorts.

Tests never touch the network (Shiller parquet) and never depend on market
history.  The synthetic generator plants exactly the conditions we want: a
long-run equity real return of ~6.8%, a bad-sequence stress version, and a
low-yield world.  The engine's behaviour against each is the study's spine.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from four_percent_rule import data  # noqa: E402


@pytest.fixture
def normal_returns():
    """Standard synthetic cohort: long-run real equity ~6.8%, 100 years."""
    df, truth = data.synthetic_cohorts(
        n_years_total=100, equity_real_mean=0.068, bad_start=False, seed=173
    )
    return df, truth


@pytest.fixture
def bad_start_returns():
    """Same mean but with the sequence-of-returns shock in the first 3 years."""
    df, truth = data.synthetic_cohorts(
        n_years_total=100, equity_real_mean=0.068, bad_start=True, seed=173
    )
    return df, truth


@pytest.fixture
def low_yield_returns():
    """Low-yield world: equity real mean 3%, bond real mean 0.5%."""
    df, truth = data.synthetic_cohorts(
        n_years_total=100, equity_real_mean=0.03, bond_real_mean=0.005,
        bad_start=False, seed=173
    )
    return df, truth
