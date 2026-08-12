"""Offline tests for the data layer — determinism, shapes, the flat-vol null flag."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vt6040 import data  # noqa: E402


def test_synthetic_deterministic(edge_world):
    frame, _ = edge_world
    f2, _ = data.synthetic_prices(seed=904, n_days=6000)
    assert np.allclose(frame.to_numpy(), f2.to_numpy())


def test_synthetic_shape_and_columns(edge_world):
    frame, truth = edge_world
    assert list(frame.columns) == ["SPY", "IEF", "BIL"]
    assert (frame > 0).all().all()          # strictly positive prices
    assert frame.index.is_monotonic_increasing
    assert truth.n_days == len(frame)


def test_regime_flag(edge_world, null_world):
    _, t_edge = edge_world
    _, t_null = null_world
    assert t_edge.has_regime is True
    assert t_null.has_regime is False       # flat vol -> nothing to read


def test_theoretical_ceiling_gt_one_only_with_regime(edge_world, null_world):
    _, t_edge = edge_world
    _, t_null = null_world
    # Cauchy-Schwarz: the perfect-foresight overlay ceiling exceeds 1 only when regimes differ.
    assert t_edge.theoretical_sharpe_gain > 1.05
    assert abs(t_null.theoretical_sharpe_gain - 1.0) < 1e-9


def test_cash_column_is_near_constant_growth(edge_world):
    frame, _ = edge_world
    cash_ret = frame["BIL"].pct_change().dropna()
    assert cash_ret.std() < 1e-9            # constant daily bill accrual
    assert cash_ret.mean() > 0


def test_no_timestamp_overflow():
    # Large n_days must stay well below the pandas ns-timestamp horizon (~year 2262).
    frame, _ = data.synthetic_prices(seed=1, n_days=9000)
    assert frame.index.max().year < 2100


def test_load_prices_empty_without_cache(tmp_path, monkeypatch):
    # Point the cache at an empty dir: load_prices must return an empty frame, never import yf.
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data, "_cache_path", lambda tk: os.path.join(str(tmp_path), f"{tk}.parquet"))
    assert data.have_real() is False
    assert data.load_prices().empty
