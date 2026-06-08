"""Shared fixtures — the offline synthetic series the whole suite asserts on.

A log-VIX-like series with **known** fixed cycles (80 and 40 sessions) buried in AR(1) red
noise. The detector must recover the injected periods and the projector must forecast their
turns; a pure red-noise twin must produce *no* significant peak. If those hold, a null on the
real VIX is a fact about the market, not a bug.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vix_cycles import data  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """(series, injected cycles) — fixed 80d & 40d cycles in red noise. Deterministic."""
    series, injected = data.synthetic_cycle(seed=0)
    return series, injected


@pytest.fixture(scope="session")
def red_noise(synth):
    """A pure AR(1) red-noise series matching the synthetic's length/variance — no cycle."""
    series, _ = synth
    return data.red_noise_like(series, rho=0.94, seed=7)
