"""Shared fixtures — deterministic synthetic factor zoos (a pure-noise null the demo
mines, and a zoo with a known planted true subset = the positive control), so tests never
touch the network and the only thing an honest correction can keep (a real factor) is
either baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tstat_threshold import data  # noqa: E402


@pytest.fixture
def null_zoo():
    """Pure-noise zoo (N=1000, T=240): every 'significant' factor is a false discovery."""
    R, is_true, truth = data.synthetic_zoo(
        n_factors=1000, n_periods=240, n_true=0, seed=839
    )
    return R, is_true, truth


@pytest.fixture
def mixture_zoo():
    """50 genuinely-priced factors (expected |t| = 4) buried in 1000 candidates — the
    positive control the corrections must keep while purging the noise."""
    R, is_true, truth = data.synthetic_zoo(
        n_factors=1000, n_periods=240, n_true=50, expected_t=4.0, seed=839
    )
    return R, is_true, truth
