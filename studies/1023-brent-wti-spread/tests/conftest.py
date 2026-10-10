"""Shared fixtures for Study 1023 — deterministic, offline, no network.

Three synthetic worlds from one generator, one knob moved at a time:

* ``planted`` — an OU log Brent/WTI ratio with a three-month half-life (``signal_strength=1``),
  calibrated to the real tape's pre-break regime: the tether the claim believes in;
* ``null_world`` — the same shocks with ``signal_strength=0``: the ratio is a random walk and
  nothing in this study may find a tether in it;
* ``broken`` — the planted tether plus one level break of the real tape's size (0.15 in logs):
  the world the claim's believers actually lived in after 2010.

A test that passes on ``planted`` and also passes on ``null_world`` is not testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cushing import data  # noqa: E402

SEED = 1023
BREAK = 0.15


@pytest.fixture(scope="session")
def planted():
    return data.synthetic_spread(signal_strength=1.0, seed=SEED)


@pytest.fixture(scope="session")
def null_world():
    return data.synthetic_spread(signal_strength=0.0, seed=SEED)


@pytest.fixture(scope="session")
def broken():
    return data.synthetic_spread(signal_strength=1.0, break_size=BREAK, seed=SEED)
