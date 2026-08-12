"""Shared fixtures — deterministic synthetic small-cap worlds (no network, no real data)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from profitable_small import data  # noqa: E402


def _as_frame(world: pd.DataFrame) -> pd.DataFrame:
    """Rename the synthetic world's legs onto the ``x_<ticker>`` schema the strategy expects."""
    return pd.DataFrame(
        {"x_CALF": world["x_quality"], "x_IWM": world["x_plain"], "x_SPY": world["x_large"],
         "r_CALF": world["r_quality"], "r_IWM": world["r_plain"], "r_SPY": world["r_large"],
         "rf": world["rf"]},
        index=world.index,
    )


@pytest.fixture(scope="session")
def edge_world():
    """A planted quality edge: profitable small caps out-earn junky small caps."""
    return _as_frame(data.synthetic_world(edge=0.4, seed=901))


@pytest.fixture(scope="session")
def null_world():
    """The null — quality and plain differ only by zero-mean junk noise (no edge)."""
    return _as_frame(data.synthetic_world(edge=0.0, seed=901))
