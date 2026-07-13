"""Shared fixtures — fully offline. The S2F curve is a pure function of the issuance schedule;
the synthetic world plants (or withholds) a valuation->return effect, so the only thing the
predictive machinery can recover is what we baked in. No network anywhere."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stock_to_flow import data  # noqa: E402


@pytest.fixture
def sf_curve():
    """The deterministic reconstructed stock-to-flow curve."""
    return data.supply_flow_daily()


@pytest.fixture
def null_world():
    """No valuation->return link (beta=0): the residual carries no forward-return information."""
    return data.synthetic_world(beta=0.0, seed=765)


@pytest.fixture
def planted_world():
    """A planted mean-reversion-to-model effect (beta=0.03): cheap predicts positive returns."""
    return data.synthetic_world(beta=0.03, seed=765)
