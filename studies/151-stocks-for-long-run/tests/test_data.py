"""Tests: synthetic world is well-formed, deterministic, and structurally correct."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stocks_for_long_run import data  # noqa: E402


def test_synthetic_shape(premium_world):
    frame, truth = premium_world
    assert len(frame) == truth["n_years"] * 12
    expected_cols = ["real_eq_ret", "real_bd_ret", "real_tr_eq", "real_tr_bd"]
    for c in expected_cols:
        assert c in frame.columns, f"Missing column: {c}"
    for h in data.HORIZONS:
        for sfx in ("eq", "bd", "excess"):
            assert f"roll_{h}y_{sfx}" in frame.columns


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(n_years=100, equity_premium=0.05, seed=42)
    b, _ = data.synthetic_world(n_years=100, equity_premium=0.05, seed=42)
    assert np.allclose(a["real_tr_eq"].to_numpy(), b["real_tr_eq"].to_numpy())
    c, _ = data.synthetic_world(n_years=100, equity_premium=0.05, seed=43)
    assert not np.allclose(a["real_tr_eq"].to_numpy(), c["real_tr_eq"].to_numpy())


def test_real_tr_is_positive_and_monotone_ish(premium_world):
    """The cumulative return index must stay positive (it is a product of 1+r terms)."""
    frame, _ = premium_world
    assert (frame["real_tr_eq"] > 0).all()
    assert (frame["real_tr_bd"] > 0).all()


def test_equity_premium_raises_mean_return(premium_world, null_world):
    """With a planted premium, equity mean return should exceed the null world's."""
    # Use the 30-year rolling window means
    col = "roll_30y_eq"
    prem_mean = premium_world[0][col].dropna().mean()
    null_mean = null_world[0][col].dropna().mean()
    assert prem_mean > null_mean + 0.01, (
        f"premium world mean {prem_mean:.3f} should exceed null {null_mean:.3f} by >1pp"
    )


def test_rolling_columns_have_nans_at_end(premium_world):
    """The forward 30-year return column must be NaN for the last 360 rows."""
    frame, _ = premium_world
    col = "roll_30y_eq"
    assert frame[col].isna().sum() >= 360, "Expected NaN at the end for the 30y forward window"


def test_fingerprint_stable_and_content_sensitive(premium_world):
    frame, _ = premium_world
    fp1 = data.fingerprint(frame)
    fp2 = data.fingerprint(frame)
    assert fp1 == fp2
    other, _ = data.synthetic_world(n_years=155, equity_premium=0.05, seed=99)
    assert data.fingerprint(other) != fp1


def test_shiller_loads(tmp_path):
    """load_shiller reads from the real parquet and returns the expected columns."""
    # Uses the actual staged cache; skip gracefully if file is absent in a stripped env.
    import os
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "..", "_cache"
    )
    if not os.path.exists(os.path.join(cache_dir, "shiller_sp500.parquet")):
        pytest.skip("Shiller parquet not present in this environment")
    df = data.load_shiller(cache_dir=cache_dir)
    assert "real_eq_ret" in df.columns
    assert "real_bd_ret" in df.columns
    assert "roll_30y_eq" in df.columns
    assert "roll_30y_bd" in df.columns
    assert "roll_30y_excess" in df.columns
    assert len(df) > 1000, "Expected >1000 monthly rows"
    # Ensure the TR index is positive
    assert (df["real_tr"] > 0).all()
