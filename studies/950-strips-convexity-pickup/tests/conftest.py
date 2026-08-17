"""Shared fixtures — deterministic synthetic tapes for Study 950 (Zero-Coupon Convexity).

Both fixtures are offline and deterministic (fixed seed 950; no network, no cache):

- ``planted`` — a two-bond world in which the long (zero-coupon) leg carries a genuine
  convexity pickup per unit of duration (``signal_strength=1``), paid for with a planted
  carry give-up. The asymmetry regression must recover a positive quadratic coefficient
  and a negative intercept.
- ``null_world`` — the same world with the convexity gap switched off
  (``signal_strength=0``): a duration-matched mix of the short leg has exactly the long
  leg's convexity, so there is no asymmetry to find and the harness must stay quiet.

The planted gap (2.5x the short leg's convexity per unit of duration) is deliberately far
larger than anything a real Treasury curve shows — these fixtures test the *detector*, not
the plausibility of the effect.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from zero_convexity import data  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    """A genuine convexity pickup on the long leg, priced with a carry give-up."""
    return data.synthetic_panel(signal_strength=1.0, seed=950)


@pytest.fixture(scope="module")
def null_world():
    """No convexity gap — the duration-matched mix is convexity-matched too (the null)."""
    return data.synthetic_panel(signal_strength=0.0, seed=950)
