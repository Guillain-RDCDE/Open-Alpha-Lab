"""Shared fixtures — deterministic synthetic panels for Study 958 (Spot ETF Basis).

Both fixtures are offline and deterministic (fixed seed 958; no network, no cache):

- ``compressed`` — a panel with a **planted compression** of the futures basis at the
  event date (``signal_strength=1.0``): the era test must recover it.
- ``uncompressed`` — the same world with the basis unchanged through the event
  (``signal_strength=0.0``): the null, on which the era test must stay quiet even
  though the *level* of the drag is large in both halves.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from etf_basis import data  # noqa: E402


@pytest.fixture
def compressed():
    """A panel whose futures basis genuinely compresses at the event date."""
    return data.synthetic_panel(signal_strength=1.0, seed=958)


@pytest.fixture
def uncompressed():
    """The null: a large but *constant* basis straight through the event date."""
    return data.synthetic_panel(signal_strength=0.0, seed=958)
