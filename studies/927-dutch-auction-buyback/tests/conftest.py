"""Shared fixtures — deterministic synthetic panels for Study 927 (Dutch Auction).

Both fixtures are offline and deterministic (fixed seed 927; no network, no cache):

- ``planted`` — 60 synthetic issuers, each with a *planted* announcement-day jump and a
  *planted* six-month post-expiry drift (``signal_strength=1``). The event study must
  recover both.
- ``null_panel`` — the same world with ``signal_strength=0``: no jump, no drift. The event
  study must stay quiet on the day-0 measure, whose cross-event *t* is correctly sized.

Panels are kept small (60 names, 900 sessions) so the whole suite runs in seconds.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dutch_auction import data  # noqa: E402


@pytest.fixture
def planted():
    """A panel with a real announcement jump and a real post-expiry drift."""
    return data.synthetic_panel(n_events=60, n_days=900, signal_strength=1.0, seed=927)


@pytest.fixture
def null_panel():
    """The pure null — no planted jump, no planted drift."""
    return data.synthetic_panel(n_events=60, n_days=900, signal_strength=0.0, seed=927)
