"""Data-layer tests — the synthetic panel's knobs, the Corwin-Schultz estimator, and the
cache-first real loader (guarded by cache presence). Offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hf_reversal import data  # noqa: E402


def test_synthetic_shape_and_determinism():
    a, ta = data.synthetic_panel(n_firms=50, n_weeks=120, reversal=0.3, bounce=0.005, seed=800)
    b, tb = data.synthetic_panel(n_firms=50, n_weeks=120, reversal=0.3, bounce=0.005, seed=800)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (120, 50)
    assert ta == tb
    assert (a > 0).all().all(), "prices must be positive"


def test_synthetic_null_has_no_planted_effect():
    px, truth = data.synthetic_panel(reversal=0.0, bounce=0.0, seed=800)
    assert truth["reversal"] == 0.0 and truth["bounce"] == 0.0
    # a pure null tape's weekly returns should be ~i.i.d.: tiny lag-1 autocorrelation
    ret = px.pct_change().dropna()
    ac = np.nanmean([ret[c].autocorr(1) for c in ret.columns])
    assert abs(ac) < 0.1, f"null lag-1 autocorr too large: {ac:.3f}"


def test_bounce_injects_negative_autocorrelation():
    """Pure bid-ask bounce must create negative lag-1 autocorrelation in observed returns."""
    px, _ = data.synthetic_panel(reversal=0.0, bounce=0.02, seed=800)
    ret = px.pct_change().dropna()
    ac = np.nanmean([ret[c].autocorr(1) for c in ret.columns])
    assert ac < -0.05, f"bounce should push lag-1 autocorr negative, got {ac:.3f}"


def test_corwin_schultz_nonnegative_and_sane():
    rng = np.random.default_rng(0)
    n = 500
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = pd.Series(mid * (1 + rng.uniform(0.002, 0.02, n)))
    low = pd.Series(mid * (1 - rng.uniform(0.002, 0.02, n)))
    s = data.corwin_schultz_daily(high, low)
    s = s.dropna()
    assert (s >= 0).all(), "CS spread estimates are floored at zero"
    assert s.mean() < 0.5, "a sane proportional spread is well under 50%"


def _have_shared_daily() -> bool:
    return data.have_real() or data._shared_daily_path() is not None


requires_panel = pytest.mark.skipif(
    not _have_shared_daily(),
    reason="no weekly cache or shared daily panel (offline CI); covered by synthetic tests",
)


@requires_panel
def test_load_real_shapes():
    close, spread = data.load_real()
    assert close.shape[0] > 200 and close.shape[1] > 100
    assert spread.shape == close.shape
    assert close.index.max() <= pd.Timestamp(data.AS_OF_LAST_WEEK)
