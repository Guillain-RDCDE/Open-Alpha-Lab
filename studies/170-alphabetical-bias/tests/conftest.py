"""Shared fixtures for Study 170 (Alphabetical-Bias).

Tests run entirely offline using the deterministic synthetic cross-section
generator.  Two panels are shared across test modules:

- ``null_panel`` — no return bias, no volume bias (the null: alphabetical order
  carries no information).
- ``biased_panel`` — a planted return premium for early-alphabet stocks (the
  positive control: the engine must detect this).
"""

import os
import sys

import pytest

# Make the study package importable when pytest is run from the study root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from alphabetical_bias import data  # noqa: E402


@pytest.fixture(scope="session")
def null_panel():
    """Synthetic cross-section with no alphabetical return or volume bias."""
    panel, truth = data.synthetic_cross(
        n_stocks=150,
        n_months=60,
        return_bias_bps=0.0,
        volume_bias=0.0,
        seed=170,
    )
    return panel, truth


@pytest.fixture(scope="session")
def biased_panel():
    """Synthetic cross-section with a planted 30 bps/month early-alphabet premium."""
    panel, truth = data.synthetic_cross(
        n_stocks=300,
        n_months=240,
        return_bias_bps=30.0,
        volume_bias=0.30,
        seed=170,
    )
    return panel, truth
