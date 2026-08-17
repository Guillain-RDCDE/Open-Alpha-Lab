"""Shared fixtures — deterministic synthetic panels for Study 917 (Stale NAV).

Two fixtures, both offline and deterministic (fixed seed 917; no network):

- ``planted`` — a five-fund panel in which yesterday's US session genuinely leaks into
  today's country return (``signal_strength=1`` → a +0.25 stale-price catch-up beta).
  The machinery MUST find it.
- ``null_panel`` — the same world with the catch-up switched off (``signal_strength=0``):
  the funds still load on the *contemporaneous* US move (beta 0.55), so the tape looks
  exactly as correlated, but there is nothing left to predict. The machinery MUST stay
  quiet.

No test reads ``studies/_cache`` — the suite is green on a fresh checkout with no cache.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stale_nav import data  # noqa: E402


@pytest.fixture
def planted():
    """A panel with a genuine +0.25 next-day catch-up from the US session."""
    return data.synthetic_panel(signal_strength=1.0, seed=917)


@pytest.fixture
def null_panel():
    """The null: same contemporaneous correlation, zero next-day catch-up."""
    return data.synthetic_panel(signal_strength=0.0, seed=917)
