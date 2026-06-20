"""Shared fixtures — deterministic synthetic small-cap tapes with a *known* amount of
pre-reconstitution run-up premium, so tests never touch the network and the only thing the
front-run rule can harvest (a planted run-up drift) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from russell_reconstitution import data  # noqa: E402

# Real cache path used for cache-gating any real-tape test (offline CI skips it).
IWM_CACHE = os.path.join(data.DEFAULT_CACHE, "bars_IWM_daily.parquet")


@pytest.fixture
def null_tape():
    """A null tape (premium_bps=0) — the front-run rule must find nothing here."""
    bars, truth = data.synthetic_daily(start_year=2000, end_year=2025, premium_bps=0.0, seed=320)
    return bars, truth


@pytest.fixture
def planted_tape():
    """A tape with a large planted run-up premium (60 bps/day) — engine should detect it."""
    bars, truth = data.synthetic_daily(start_year=2000, end_year=2025, premium_bps=60.0, seed=320)
    return bars, truth
