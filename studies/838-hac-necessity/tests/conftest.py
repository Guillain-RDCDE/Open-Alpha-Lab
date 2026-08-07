"""Shared fixtures — deterministic synthetic null worlds (no network, no real data).

The null matrices have a true mean of ZERO by construction (so any rejection is a false
positive) but strong serial correlation (an overlapping window / an AR(1) dial), which is
exactly what makes the naive i.i.d. t-stat misbehave.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hac_necessity import data  # noqa: E402


@pytest.fixture(scope="session")
def overlap_null():
    """Monte-Carlo matrix of 21-day overlapping null paths (mean 0, MA(20) autocorrelation)."""
    return data.overlap_matrix(400, 2000, window=21, seed=838, daily_vol=0.01, mean=0.0)


@pytest.fixture(scope="session")
def iid_null():
    """The no-autocorrelation control: window=1 is plain i.i.d. noise (mean 0)."""
    return data.overlap_matrix(400, 2000, window=1, seed=838, daily_vol=0.01, mean=0.0)


@pytest.fixture(scope="session")
def ar1_null():
    """Monte-Carlo matrix of AR(1) null paths, rho=0.6 (mean 0)."""
    return data.ar1_matrix(400, 2000, rho=0.6, seed=838, daily_vol=0.01, mean=0.0)
