"""Shared fixtures — deterministic synthetic minute tapes with the post-breakout answer baked in, so
tests never touch the network and the win rate the bracket should recover is known in advance: a
**null** tape (no continuation → coin flip), a **continuation** tape (breakouts follow through →
win rate above 0.5), an **exhaustion** tape (breakouts fade → below 0.5), and a **grind-gated** tape
(continuation only after a calm approach → something for the staircase filter to find)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from glass_ceiling import data


@pytest.fixture
def null_tape():
    """No post-breakout drift — a fresh high carries no information. The coin-flip baseline."""
    return data.synthetic_intraday(n_bars=90_000, cont_drift=0.0, seed=17)


@pytest.fixture
def cont_tape():
    """Genuine continuation: breakouts follow through, so the long bracket should win > 50%."""
    return data.synthetic_intraday(n_bars=90_000, cont_drift=0.0008, cont_window=20, seed=17)


@pytest.fixture
def fade_tape():
    """Exhaustion: breakouts fade, so the long bracket should win < 50% (the buy-the-high trap)."""
    return data.synthetic_intraday(n_bars=90_000, cont_drift=-0.0008, cont_window=20, seed=17)


@pytest.fixture
def grind_tape():
    """Continuation gated on a calm approach — only grind-preceded breakouts follow through.

    Drift is deliberately mild so the grind/no-grind win-rate split is visible but not saturated;
    ``lookback`` matches the filter's default window so the generator's grind gate and the filter's
    grind score read the same bars."""
    return data.synthetic_intraday(n_bars=90_000, cont_drift=0.0006, cont_window=20, lookback=20,
                                   cont_requires_grind=True, seed=17)
