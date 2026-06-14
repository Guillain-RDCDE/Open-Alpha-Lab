"""Tests for data.py — synthetic tape is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_capture import data  # noqa: E402


def test_synthetic_shape_and_columns(efficient):
    events, truth = efficient
    assert len(events) == truth["n_events"]
    expected_cols = [
        "ex_date", "ticker", "div", "price_before", "price_ex",
        "price_after", "drop", "drop_ratio", "ret_gross", "ret_net",
    ]
    for col in expected_cols:
        assert col in events.columns, f"missing column: {col}"


def test_synthetic_prices_positive(efficient):
    events, _ = efficient
    assert (events["price_before"] > 0).all()
    assert (events["price_ex"] > 0).all()
    assert (events["price_after"] > 0).all()
    assert (events["div"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_events(n_events=100, efficiency=1.0, seed=10)
    b, _ = data.synthetic_events(n_events=100, efficiency=1.0, seed=10)
    assert np.allclose(a["div"].to_numpy(), b["div"].to_numpy())
    c, _ = data.synthetic_events(n_events=100, efficiency=1.0, seed=99)
    assert not np.allclose(a["div"].to_numpy(), c["div"].to_numpy())


def test_drop_ratio_near_one_when_fully_efficient(efficient):
    """With efficiency=0.0 (price drops by full div), mean drop ratio should be near 1.0."""
    events, _ = efficient
    mean_ratio = events["drop_ratio"].mean()
    # Noise from daily vol; allow ±0.5 tolerance on a noisy synthetic tape.
    assert 0.5 < mean_ratio < 1.5, f"expected drop_ratio near 1.0, got {mean_ratio:.3f}"


def test_drop_ratio_lower_when_inefficient(inefficient, efficient):
    """With efficiency=0.5, the mean drop ratio is ~0.5 < 1.0 (the efficient case)."""
    eff_ratio = efficient[0]["drop_ratio"].mean()
    ineff_ratio = inefficient[0]["drop_ratio"].mean()
    # The inefficient tape (efficiency=0.5) has a smaller drop ratio than the efficient (efficiency=0.0).
    assert ineff_ratio < eff_ratio, (
        f"inefficient drop_ratio ({ineff_ratio:.3f}) should be < efficient ({eff_ratio:.3f})"
    )


def test_ret_net_less_than_ret_gross(efficient):
    """ret_net must always be ret_gross minus the fixed cost (5 bps default)."""
    events, truth = efficient
    diff = (events["ret_gross"] - events["ret_net"]).to_numpy()
    expected_cost = truth["cost_bps"] * 1e-4
    assert np.allclose(diff, expected_cost, atol=1e-10), "ret_net != ret_gross - cost"


def test_fetch_events_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_events(tickers=["NOPE"], fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(efficient):
    events, _ = efficient
    assert data.fingerprint(events) == data.fingerprint(events)
    other, _ = data.synthetic_events(n_events=200, efficiency=1.0, seed=999)
    assert data.fingerprint(events) != data.fingerprint(other)
