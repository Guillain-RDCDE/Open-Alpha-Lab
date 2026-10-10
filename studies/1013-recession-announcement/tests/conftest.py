"""Shared fixtures for Study 1013 — deterministic, offline, no network.

The synthetic world carries an NBER-style calendar (peaks, troughs and the two announcements
of each cycle) on a monthly market. At ``signal_strength = 1`` it plants, at known sizes, a bear
market that *leads* each synthetic NBER peak and trough by five months and a +15% abnormal
return over the year after every announcement. At ``0`` the calendar is the same kind of object
and carries no information at all — the matched null.

A detector that fires on both worlds is not detecting anything, so every strategy test that
claims to find something on ``planted`` is paired with one that finds nothing on ``null``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nberclock import data  # noqa: E402

SEED = 1013


@pytest.fixture(scope="session")
def planted():
    """The world with the lead and the announcement premium planted (signal_strength=1)."""
    return data.synthetic_world(signal_strength=1.0, seed=SEED)


@pytest.fixture(scope="session")
def null():
    """The matched null: i.i.d. returns, an uninformative calendar (signal_strength=0)."""
    return data.synthetic_world(signal_strength=0.0, seed=SEED)
