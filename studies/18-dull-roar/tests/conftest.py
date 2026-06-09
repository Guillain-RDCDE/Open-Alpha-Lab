"""Shared fixtures — deterministic synthetic universes, so tests never touch the network and the
quantities the diagnostics hunt are baked in: an **anomaly** panel with a flat security-market line
(low-beta names carry positive alpha, so the vol sort earns a beta-neutral alpha) and a **null** panel
(textbook CAPM, alpha=0 everywhere, so the sort must add nothing on a beta-adjusted basis)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dull_roar import data

SEED = 18


@pytest.fixture
def anomaly():
    """~4000 bars, 200 names, a baked flat-SML low-vol anomaly (the effect the sort should recover)."""
    return data.synthetic_panel(sml_slope=0.00018, seed=SEED)


@pytest.fixture
def null():
    """~4000 bars, 200 names, a fair-CAPM universe — alpha=0 for all, so a vol sort earns no alpha."""
    return data.synthetic_panel(sml_slope=0.0, seed=SEED)


@pytest.fixture
def anomaly_panel(anomaly):
    return anomaly[0]


@pytest.fixture
def anomaly_market(anomaly):
    return anomaly[1]


@pytest.fixture
def null_panel(null):
    return null[0]


@pytest.fixture
def null_market(null):
    return null[1]
