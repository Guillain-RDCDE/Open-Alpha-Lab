"""Shared fixtures — deterministic synthetic hype-cycle tapes with a *known* behaviour
gap (or none), so tests never touch the network and the only thing the dollar-weighting
machinery can detect (performance-chasing flows) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ark_innovation import data  # noqa: E402


@pytest.fixture
def steady_grower():
    """A steady-growth fund with flat flows (bust=0, chase=0) — the average dollar earns
    roughly the buy-and-hold return, so the behaviour gap is ~0 (the null)."""
    frame, truth = data.synthetic_hype_cycle(n_months=180, bust=0.0, chase=0.0, seed=334)
    return frame, truth


@pytest.fixture
def boom_bust():
    """A boom-bust fund with performance-chasing flows (bust>0, chase>0) — money piles in
    near the peak and rides the bust down, so the dollar-weighted return lags the
    time-weighted one by a clearly positive behaviour gap (the positive control)."""
    frame, truth = data.synthetic_hype_cycle(n_months=180, bust=0.04, chase=15.0, seed=334)
    return frame, truth


@pytest.fixture
def trending_daily():
    """A daily tape with strong planted return persistence and zero drift — a momentum
    overlay should clear the inference bar harvesting it."""
    return data.synthetic_daily(n_days=3000, momentum=0.20, drift=0.0, seed=334)


@pytest.fixture
def random_daily():
    """A daily tape with no momentum and zero drift — the momentum overlay has nothing to
    harvest (the null)."""
    return data.synthetic_daily(n_days=3000, momentum=0.0, drift=0.0, seed=334)
