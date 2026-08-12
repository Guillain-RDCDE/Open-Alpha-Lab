"""Shared fixtures — deterministic synthetic vol-clustered worlds (no network, no real data)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from managed_vol import data  # noqa: E402


def _as_frame(r):
    """Wrap a synthetic excess-return array as the ex DataFrame (cash = 0%)."""
    s = pd.Series(np.asarray(r, dtype=float))
    return pd.DataFrame({"spy": s, "cash": 0.0, "spy_excess": s})


@pytest.fixture(scope="session")
def null_world():
    """Risk fully priced (mu_t ~ sig_t^2): vol targeting must earn NO timing alpha."""
    r, sig = data.synthetic_world(n_days=4000, disconnect=0.0, seed=898)
    return _as_frame(r), sig


@pytest.fixture(scope="session")
def planted_world():
    """Planted leverage effect (mean falls as variance rises): the overlay MUST light up."""
    r, sig = data.synthetic_world(n_days=4000, disconnect=2.0, seed=898)
    return _as_frame(r), sig
