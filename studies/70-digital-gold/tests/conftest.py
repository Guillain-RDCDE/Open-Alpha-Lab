"""Shared fixtures — deterministic synthetic BTC/stock/gold worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from digital_gold import data


@pytest.fixture(scope="session")
def risk_world():
    """BTC loads on the stock factor — a high-beta risk asset (the real-world finding)."""
    return data.synthetic_world(btc_stock_beta=1.5, seed=70)


@pytest.fixture(scope="session")
def haven_world():
    """BTC uncorrelated with stocks — a true 'digital gold' (the counterfactual)."""
    return data.synthetic_world(btc_stock_beta=0.0, seed=70)
