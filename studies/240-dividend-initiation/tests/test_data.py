"""Synthetic panel is deterministic; masks are boolean; the real loader is cache-safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_initiation import data  # noqa: E402

CACHE_SENTINEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "di_AAPL_px.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_SENTINEL),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic panel — determinism and structure
# ---------------------------------------------------------------------------
def test_synthetic_panel_is_deterministic():
    import pandas as pd  # noqa: F401

    a = data.synthetic_panel(initiation_premium=0.04, seed=240)
    b = data.synthetic_panel(initiation_premium=0.04, seed=240)
    assert a["initiation_mask"].equals(b["initiation_mask"])
    assert np.allclose(
        a["fwd_ret"].to_numpy(dtype=float),
        b["fwd_ret"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_synthetic_panel_different_seeds_differ():
    a = data.synthetic_panel(seed=240)
    b = data.synthetic_panel(seed=99)
    assert not a["fwd_ret"].equals(b["fwd_ret"])


def test_synthetic_panel_shape(real_world):
    p = real_world
    n_years = p["truth"]["n_years"]
    n_names = p["truth"]["n_names"]
    assert p["initiation_mask"].shape == (n_years, n_names)
    assert p["fwd_ret"].shape == (n_years, n_names)
    assert p["control_mask"].shape == (n_years, n_names)


def test_synthetic_initiation_mask_is_boolean(real_world):
    """Initiation mask must contain only booleans."""
    assert real_world["initiation_mask"].dtypes.apply(lambda d: d == bool).all()


def test_synthetic_initiation_and_control_exclusive(real_world):
    """In any given year-ticker cell, initiation and control cannot both be True."""
    overlap = real_world["initiation_mask"] & real_world["control_mask"]
    assert not overlap.any().any(), "initiation_mask and control_mask overlap — logic error"


def test_synthetic_ew_ret_is_mean_of_fwd(real_world):
    """ew_ret must equal the row mean of fwd_ret."""
    p = real_world
    ew_recomputed = p["fwd_ret"].mean(axis=1)
    assert np.allclose(p["ew_ret"].to_numpy(), ew_recomputed.to_numpy(), equal_nan=True)


def test_synthetic_premium_shifts_initiator_returns():
    """Planting a premium should make initiator mean > null mean."""
    from dividend_initiation import strategy as st

    p_real = data.synthetic_panel(initiation_premium=0.15, seed=240, n_years=30)
    p_null = data.synthetic_panel(initiation_premium=0.00, seed=240, n_years=30)
    gr = st.arm_returns(p_real["fwd_ret"], p_real["initiation_mask"]).mean()
    gn = st.arm_returns(p_null["fwd_ret"], p_null["initiation_mask"]).mean()
    assert gr > gn, "planted premium is not visible in initiator arm"


def test_fingerprint_stable_and_sensitive():
    p = data.synthetic_panel(seed=240)
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
def test_real_panel_loads_and_has_events():
    """Real panel loads without error and contains at least some initiation events."""
    p = data.load_real_panel(fetch=False)
    assert not p["initiation_mask"].empty
    assert p["initiation_mask"].any().any(), "no initiation events found in real panel"


@requires_cache
def test_real_panel_fwd_ret_not_all_nan():
    p = data.load_real_panel(fetch=False)
    assert p["fwd_ret"].notna().any().any()


@requires_cache
def test_real_panel_fingerprint_is_12_chars():
    p = data.load_real_panel(fetch=False)
    fp = data.fingerprint(p)
    assert isinstance(fp, str) and len(fp) == 12
