"""Shared fixtures — deterministic synthetic tapes for Study 945 (The Hidden Financing).

Two fixtures, both offline and deterministic (fixed seed 945; no network):

- ``planted`` — synthetic 2x and 3x wrappers built from a *known* financing rate of
  ``benchmark + 75 bp`` and a *known* 0.90% expense ratio (``signal_strength=1``). The
  estimator has to recover the 75 bp.
- ``null_tape`` — the same world with the wrappers financed at exactly the benchmark rate
  and charging nothing (``signal_strength=0``). The estimator has to report ~0 bp.

Both worlds carry the identical index path, tracking noise and rate cycle for a given
seed, so the *difference* between the two recovered spreads is the planted effect with no
sampling noise at all — which is what makes the recovery test exact rather than
approximate.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lev_financing import data  # noqa: E402


@pytest.fixture
def planted():
    """Synthetic wrappers financed at benchmark + 75 bp, charging a 0.90% fee."""
    return data.synthetic_panel(signal_strength=1.0, seed=945)


@pytest.fixture
def null_tape():
    """The null: wrappers financed at exactly the benchmark rate, charging nothing."""
    return data.synthetic_panel(signal_strength=0.0, seed=945)
