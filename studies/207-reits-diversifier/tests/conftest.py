"""Shared fixtures — deterministic synthetic price frames with a *known* amount of
REIT diversification benefit, so tests never touch the network and the only thing
the analysis can detect (genuine independent income) is either present or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reits_diversifier import data  # noqa: E402

# Paths for real-data cache guards
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PANEL_PATH = os.path.join(_REPO_ROOT, "_cache", "cross_asset_etfs.parquet")
_SHILLER_PATH = os.path.join(_REPO_ROOT, "_cache", "shiller_sp500.parquet")

requires_cache = pytest.mark.skipif(
    not os.path.exists(_PANEL_PATH),
    reason="cross_asset_etfs cache absent (offline CI); covered by synthetic tests",
)
requires_shiller = pytest.mark.skipif(
    not os.path.exists(_SHILLER_PATH),
    reason="shiller_sp500 cache absent (offline CI); covered by synthetic tests",
)


@pytest.fixture
def null_world():
    """A synthetic world where REIT == pure equity-beta (diversification=0).
    The REIT sleeve should NOT improve the 60/40 in expectation here."""
    prices, truth = data.synthetic_portfolio(n_years=20, diversification=0.0, seed=207)
    return prices, truth


@pytest.fixture
def diversified_world():
    """A synthetic world where REIT has genuine independent income (diversification=1).
    The REIT sleeve SHOULD improve the benchmark portfolio here."""
    prices, truth = data.synthetic_portfolio(n_years=20, diversification=1.0, seed=207)
    return prices, truth
