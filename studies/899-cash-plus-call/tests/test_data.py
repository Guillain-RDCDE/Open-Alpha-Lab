"""Offline tests for the data layer — determinism, shapes, the bear/convex flags, no overflow."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_call import data  # noqa: E402


def test_synthetic_deterministic(bear_world):
    frame, _ = bear_world
    f2, _ = data.synthetic_prices(seed=899, n_days=2500, drift=-0.0004, sigma=0.016)
    assert np.allclose(frame.to_numpy(), f2.to_numpy())


def test_synthetic_shape_and_columns(bear_world):
    frame, truth = bear_world
    assert list(frame.columns) == ["SPY", "BIL", "IRX"]
    assert (frame[["SPY", "BIL"]] > 0).all().all()      # strictly positive prices
    assert frame.index.is_monotonic_increasing
    assert truth.n_days == len(frame)


def test_irx_is_a_rate_in_percent(bear_world):
    frame, truth = bear_world
    # IRX is a near-constant rate in PERCENT (^IRX convention), equal to the cash rate * 100.
    assert np.allclose(frame["IRX"].to_numpy(), truth.cash_rate_ann * 100.0)
    assert frame["IRX"].mean() > 1.0                    # a few percent, not a fraction


def test_bear_and_convex_flags(bear_world, calm_world, jump_world):
    _, t_bear = bear_world
    _, t_calm = calm_world
    _, t_jump = jump_world
    assert t_bear.is_bear is True
    assert t_calm.is_bear is False
    assert t_jump.has_convex_tape is True
    assert t_bear.has_convex_tape is False              # no up-jumps in the plain bear world


def test_cash_column_is_near_constant_growth(bear_world):
    frame, _ = bear_world
    cash_ret = frame["BIL"].pct_change().dropna()
    assert cash_ret.std() < 1e-9                        # constant daily bill accrual
    assert cash_ret.mean() > 0


def test_up_jumps_inject_large_up_days(jump_world, calm_world):
    fj, _ = jump_world
    fc, _ = calm_world
    # the jump world has genuine large up days (> +5%); the calm world does not.
    assert fj["SPY"].pct_change().max() > 0.05
    assert fc["SPY"].pct_change().max() < 0.05


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
