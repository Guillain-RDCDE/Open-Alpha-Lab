"""Synthetic panel is deterministic; streaks are non-negative integers; the real loader is
cache-safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_growth import data  # noqa: E402

CACHE_SENTINEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "dg_AAPL_px.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_SENTINEL),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_is_deterministic():
    a = data.synthetic_panel(growth_premium=0.03, seed=201)
    b = data.synthetic_panel(growth_premium=0.03, seed=201)
    import pandas as pd
    assert a["streaks"].equals(b["streaks"])
    assert np.allclose(
        a["fwd_ret"].to_numpy(dtype=float),
        b["fwd_ret"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_synthetic_panel_different_seeds_differ():
    a = data.synthetic_panel(seed=201)
    b = data.synthetic_panel(seed=99)
    assert not a["fwd_ret"].equals(b["fwd_ret"])


def test_synthetic_panel_shape(real_world):
    p = real_world
    assert not p["streaks"].empty
    assert p["streaks"].shape == p["fwd_ret"].shape
    assert set(p["streaks"].columns) == set(p["fwd_ret"].columns)
    n_years = p["truth"]["n_years"]
    n_names = p["truth"]["n_names"]
    assert p["streaks"].shape == (n_years, n_names)


def test_synthetic_streaks_non_negative(real_world):
    """Streaks are non-negative integers — a name can't have negative consecutive raises."""
    assert (real_world["streaks"] >= 0).all().all()


def test_synthetic_ew_ret_is_mean_of_fwd(real_world):
    """ew_ret must equal the row mean of fwd_ret (that is what 'equal weight' means)."""
    p = real_world
    import numpy as np
    ew_recomputed = p["fwd_ret"].mean(axis=1)
    assert np.allclose(p["ew_ret"].to_numpy(), ew_recomputed.to_numpy(), equal_nan=True)


def test_synthetic_growth_premium_shifts_grower_returns():
    """Planting a premium should make grower mean > null mean (the premium is detectable)."""
    p_real = data.synthetic_panel(growth_premium=0.05, seed=201)
    p_null = data.synthetic_panel(growth_premium=0.00, seed=201)
    from dividend_growth import strategy as st
    grower_mask_r = st.classify_growers(p_real["streaks"], streak_years=3)
    grower_mask_n = st.classify_growers(p_null["streaks"], streak_years=3)
    gr = st.arm_returns(p_real["fwd_ret"], grower_mask_r).mean()
    gn = st.arm_returns(p_null["fwd_ret"], grower_mask_n).mean()
    assert gr > gn + 0.01  # planted premium is visible


def test_fingerprint_stable_and_sensitive():
    p = data.synthetic_panel(seed=201)
    q = data.synthetic_panel(seed=99)
    fp1 = data.fingerprint(p)
    fp2 = data.fingerprint(p)
    fp3 = data.fingerprint(q)
    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 12


# ---------------------------------------------------------------------------
# Real loader — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_panel_loads_and_has_sensible_shape():
    """Real panel loads without error and has positive-streak rows for major payers."""
    p = data.load_real_panel(fetch=False)
    assert not p["streaks"].empty
    assert not p["fwd_ret"].empty
    # At least one year-ticker pair has a streak > 0 (it would be alarming if no stock
    # ever raised its dividend).
    assert (p["streaks"] > 0).any().any()


@requires_cache
def test_real_panel_fwd_ret_not_all_nan():
    p = data.load_real_panel(fetch=False)
    assert p["fwd_ret"].notna().any().any()


@requires_cache
def test_real_panel_fingerprint_is_12_chars():
    p = data.load_real_panel(fetch=False)
    fp = data.fingerprint(p)
    assert isinstance(fp, str) and len(fp) == 12
