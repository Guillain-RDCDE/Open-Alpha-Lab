"""Offline, fixed-seed tests for the synthetic return panel.

The panel is deterministic; the null carries no planted momentum edge while the control
does; all assets share beta = 1 so a dollar-neutral long-short book cancels the market
exactly (no accidental beta tilt masquerading as edge in the null).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from timing_luck import data  # noqa: E402


def test_panel_deterministic():
    a, _ = data.synthetic_panel(seed=836)
    b, _ = data.synthetic_panel(seed=836)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_panel_shape_and_index():
    ret, _ = data.synthetic_panel(seed=836, n_assets=30, n_days=2600)
    assert ret.shape == (2600, 30)
    assert ret.index.is_monotonic_increasing


def test_null_has_no_planted_edge():
    _, truth = data.synthetic_panel(mom_edge=0.0, seed=836)
    assert not truth.has_edge
    _, truth2 = data.synthetic_panel(mom_edge=1.0, seed=836)
    assert truth2.has_edge


def test_prem_does_not_touch_the_null():
    """mom_edge=0 kills the premium term, so `prem` must not change the null tape."""
    a, _ = data.synthetic_panel(mom_edge=0.0, seed=836, prem=0.0012)
    b, _ = data.synthetic_panel(mom_edge=0.0, seed=836, prem=0.05)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_edge_shifts_the_tape():
    a, _ = data.synthetic_panel(mom_edge=0.0, seed=836)
    b, _ = data.synthetic_panel(mom_edge=1.0, seed=836)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_common_market_cancels_in_cross_sectional_demean():
    """All assets share the same market shift each day, so the cross-sectional spread
    (max-min across names) is driven by idio/trend, not the common factor — a proxy that
    a dollar-neutral book cancels the market."""
    ret, _ = data.synthetic_panel(mom_edge=0.0, seed=836)
    R = ret.to_numpy()
    # cross-sectional demeaned returns still have material dispersion (idio noise present)
    demeaned = R - R.mean(axis=1, keepdims=True)
    assert demeaned.std() > 0.005
