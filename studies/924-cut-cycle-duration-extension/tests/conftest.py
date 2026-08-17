"""Shared fixtures — deterministic synthetic tapes for Study 924 (First Cut).

Both fixtures are offline and deterministic (fixed seed 924; no network, no cache):

- ``planted`` — a duration / front-end / cash tape with five planted "first cut" dates
  after which duration earns a real six-month excess return (``signal_strength=1``).
- ``null_world`` — the same generator with the planted effect switched off
  (``signal_strength=0``): the event study must find nothing.

A third fixture, ``null_panel``, returns twelve independent null worlds, because with
five events per world a single world's *t* is far too noisy to test against.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from first_cut import data  # noqa: E402


@pytest.fixture
def planted():
    """A tape with a genuine, planted post-first-cut duration rally."""
    return data.synthetic_daily(signal_strength=1.0, seed=924)


@pytest.fixture
def null_world():
    """The same tape with no event effect at all — the null."""
    return data.synthetic_daily(signal_strength=0.0, seed=924)


@pytest.fixture
def null_panel():
    """Twelve independent null worlds — the honest way to calibrate a five-event test."""
    return data.synthetic_panel(n_worlds=12, signal_strength=0.0, base_seed=924)


@pytest.fixture
def planted_panel():
    """Twelve independent planted worlds — the power side of the same calibration."""
    return data.synthetic_panel(n_worlds=12, signal_strength=1.0, base_seed=924)
