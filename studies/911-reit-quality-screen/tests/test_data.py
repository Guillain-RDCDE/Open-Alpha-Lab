"""Offline, fixed-seed tests for the Study 911 data layer (synthetic world + cache gating)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reit_quality import data  # noqa: E402


def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(edge_ann=0.03, seed=911, n_months=228)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_world_shape_and_columns(edge_world):
    assert list(edge_world.columns) == ["BROAD", "QUAL", "TRAP", "CASH"]
    assert len(edge_world) == 228
    # A monthly PeriodIndex-derived timestamp index that never overflows the ns horizon.
    assert isinstance(edge_world.index, pd.DatetimeIndex)
    assert edge_world.index.max().year < 2262


def test_planted_edge_present(edge_world, null_world):
    # QUAL out-drifts BROAD when an edge is planted, and matches it when it is not.
    edge_gap = edge_world["QUAL"].mean() - edge_world["BROAD"].mean()
    null_gap = null_world["QUAL"].mean() - null_world["BROAD"].mean()
    assert edge_gap > 0                       # planted +3%/yr edge shows through
    assert abs(null_gap) < edge_gap / 3.0     # null gap is small vs the planted edge


def test_trap_leg_structurally_worse(edge_world):
    # The TRAP sleeve carries a negative drag -> lower mean than the broad index.
    assert edge_world["TRAP"].mean() < edge_world["BROAD"].mean()


def test_cache_gating_flag():
    # have_real must be a pure existence check on a bogus path (no network side effect).
    assert data.have_real(path=os.path.join(data.CACHE_DIR, "does_not_exist.parquet")) is False


def test_roles_are_tickers():
    for role in [data.BROAD, data.QUALITY, data.TRAP, data.EQUITY, data.CASH]:
        assert role in data.TICKERS
    assert all(t in data.TICKERS for t in data.QUALITY_BLEND)
