"""Shared fixtures — deterministic synthetic tapes for Study 939 (DRIP or Sweep).

Every fixture is offline and deterministic (fixed seed 939; no network, no cache).
The tapes are short on purpose: the simulator walks the path day by day, so a 12-year
bench keeps the suite quick while still spanning ~48 distribution events.

- ``planted`` — a fat equity premium over cash (``signal_strength=1``): parking a
  distribution in T-bills genuinely costs money, so the DRIP arm must win.
- ``null`` — the fund drifts at exactly the cash rate (``signal_strength=0``):
  parking costs nothing, so the DRIP-minus-sweep gap must be ~0.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from drip_sweep import data  # noqa: E402

N_YEARS = 12


@pytest.fixture(scope="session")
def planted():
    """A tape with a real premium over cash — the DRIP arm must win."""
    return data.synthetic_daily(n_years=N_YEARS, signal_strength=1.0, seed=939)


@pytest.fixture(scope="session")
def null():
    """A tape drifting at exactly the cash rate — the gap must vanish."""
    return data.synthetic_daily(n_years=N_YEARS, signal_strength=0.0, seed=939)
