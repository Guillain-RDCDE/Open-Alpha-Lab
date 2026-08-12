"""Shared fixtures — deterministic synthetic January-seasonal panels (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dry_january import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted January abnormal-return seasonal (edge > 0) the detector must recover."""
    return data.synthetic_panel(edge=0.05, seed=849)


@pytest.fixture(scope="session")
def null_world():
    """The null — a common market factor + noise, but NO January seasonal (edge = 0)."""
    return data.synthetic_panel(edge=0.0, seed=849)
