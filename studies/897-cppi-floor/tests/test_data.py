"""Offline tests for the data layer — determinism, shapes, the bear/gap flags, no overflow."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cppi import data  # noqa: E402


def test_synthetic_deterministic(bear_world):
    frame, _ = bear_world
    f2, _ = data.synthetic_prices(seed=897, n_days=2500, drift=-0.0004, sigma=0.016)
    assert np.allclose(frame.to_numpy(), f2.to_numpy())


def test_synthetic_shape_and_columns(bear_world):
    frame, truth = bear_world
    assert list(frame.columns) == ["SPY", "IEF", "BIL"]
    assert (frame > 0).all().all()          # strictly positive prices
    assert frame.index.is_monotonic_increasing
    assert truth.n_days == len(frame)


def test_bear_and_gap_flags(bear_world, calm_world, gap_world):
    _, t_bear = bear_world
    _, t_calm = calm_world
    _, t_gap = gap_world
    assert t_bear.is_bear is True
    assert t_calm.is_bear is False
    assert t_gap.has_gap_risk is True
    assert t_bear.has_gap_risk is False     # no gaps in the plain bear world


def test_cash_column_is_near_constant_growth(bear_world):
    frame, _ = bear_world
    cash_ret = frame["BIL"].pct_change().dropna()
    assert cash_ret.std() < 1e-9            # constant daily bill accrual
    assert cash_ret.mean() > 0


def test_gaps_inject_large_down_days(gap_world, calm_world):
    fg, _ = gap_world
    fc, _ = calm_world
    # The gap world has genuine crash days (< −20%); the calm world does not.
    assert fg["SPY"].pct_change().min() < -0.20
    assert fc["SPY"].pct_change().min() > -0.05


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
