"""Shared fixtures — deterministic synthetic panels for Study 928 (Odd-Lot Priority).

Two fixtures, both offline and deterministic (fixed seed 928; no network, no cache):

- ``planted`` — a panel of synthetic issuers each carrying a *real* announcement run-up
  and a *real* post-expiry give-back (``signal_strength=1``): the world in which the
  odd-lot round trip is exactly the trade the folklore describes.
- ``null_panel`` — the same panel machinery with nothing planted
  (``signal_strength=0``): the odd-lot estimator must find no run-up and no give-back.

The whole suite runs with NO ``studies/_cache`` present, so CI is green on a fresh
checkout.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from odd_lot import data  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    """A panel with a planted run-up and a planted post-expiry give-back."""
    return data.synthetic_panel(n_events=90, signal_strength=1.0, seed=928)


@pytest.fixture(scope="module")
def null_panel():
    """The pure null: no run-up, no give-back — only market and idiosyncratic noise."""
    return data.synthetic_panel(n_events=90, signal_strength=0.0, seed=928)
