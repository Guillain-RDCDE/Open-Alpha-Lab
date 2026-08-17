"""Shared fixtures — deterministic synthetic panels for Study 929 (Rights Offering).

Two fixtures, both offline and deterministic (fixed seeds; no network, no cache):

- ``planted`` — a panel with a full-size rights-offering effect (``signal_strength=1``):
  an announcement drop, a negative drift across the subscription period and a bounce
  after expiry, scaled up for the "deep discount" band. The event study must recover it.
- ``null_panel`` — the same panel machinery with ``signal_strength=0``: the anchors are
  arbitrary sessions with no planted effect. The event study must stay quiet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rights_offering import data  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    """A synthetic panel carrying a full-size planted rights-offering effect."""
    return data.synthetic_panel(signal_strength=1.0, seed=929)


@pytest.fixture(scope="module")
def null_panel():
    """The same panel with no planted effect at all (the null)."""
    return data.synthetic_panel(signal_strength=0.0, seed=929)
