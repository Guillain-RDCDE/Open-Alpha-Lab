"""Shared fixtures for Study 549 (Spotify-Mood). Deterministic & offline.

Every fixture uses the seeded synthetic generator only — never the network or the real-market
cache — so CI passes without any cached data.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from spotify_mood import data  # noqa: E402


@pytest.fixture
def null_frame():
    frame, truth = data.synthetic_series(n_months=180, predictive_beta=0.0, seed=549)
    return frame, truth


@pytest.fixture
def edge_frame():
    frame, truth = data.synthetic_series(n_months=180, predictive_beta=0.02, seed=549)
    return frame, truth
