"""The synthetic panel is well-formed and deterministic; the real loader is cache- and
opt-in-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dartboard_portfolio import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
PANEL_CACHE = os.path.join(CACHE, "dartboard_panel.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(null_panel):
    rets, caps, truth = null_panel
    assert rets.shape == (truth["n_periods"], truth["n_assets"])
    assert len(caps) == truth["n_assets"]
    assert list(rets.columns) == list(caps.index)
    assert np.isfinite(rets.to_numpy()).all()


def test_caps_are_positive_and_normalised(null_panel):
    _, caps, _ = null_panel
    assert (caps > 0).all()
    assert abs(caps.sum() - 1.0) < 1e-9


def test_caps_are_power_law_concentrated(null_panel):
    """A few names hold most of the cap weight — the shape that makes cap-weighting bite."""
    _, caps, _ = null_panel
    n = len(caps)
    top10 = caps.sort_values(ascending=False).head(10).sum()
    # Top 10 of 80 hold ~45% — far above the 12.5% an equal-cap universe would give.
    assert top10 > 3.0 * (10.0 / n)


def test_synthetic_is_deterministic():
    a, ca, _ = data.synthetic_panel(n_periods=120, n_assets=40, size_premium=0.003, seed=7)
    b, cb, _ = data.synthetic_panel(n_periods=120, n_assets=40, size_premium=0.003, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ca.to_numpy(), cb.to_numpy())
    c, _, _ = data.synthetic_panel(n_periods=120, n_assets=40, size_premium=0.003, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_size_premium_lifts_small_caps():
    """With a positive size premium the smallest names out-earn the largest on average."""
    rets, caps, truth = data.synthetic_panel(n_periods=600, n_assets=60,
                                             size_premium=0.006, seed=350)
    order = caps.sort_values()                       # smallest first
    small = rets[order.index[:10]].mean().mean()
    large = rets[order.index[-10:]].mean().mean()
    assert small > large


def test_no_size_premium_is_flat():
    """At size_premium=0 small and large names have ~equal expected returns."""
    rets, caps, _ = data.synthetic_panel(n_periods=2000, n_assets=60,
                                         size_premium=0.0, seed=350)
    order = caps.sort_values()
    small = rets[order.index[:10]].mean().mean()
    large = rets[order.index[-10:]].mean().mean()
    assert abs(small - large) < 5e-4


def test_index_uses_period_range_not_overflow():
    """The decorative monthly index is built without overflowing ns-Timestamps."""
    rets, _, _ = data.synthetic_panel(n_periods=240, n_assets=10, seed=1)
    assert rets.index.is_monotonic_increasing
    assert rets.index[0].year >= 1900


def test_fingerprint_stable_and_sensitive(null_panel):
    rets, _, _ = null_panel
    assert data.fingerprint(rets) == data.fingerprint(rets)
    other, _, _ = data.synthetic_panel(n_periods=240, n_assets=80, size_premium=0.0, seed=99)
    assert data.fingerprint(rets) != data.fingerprint(other)


# ---- real loader: opt-in guard + cache contract ----------------------------
def test_load_real_requires_survivorship_optin():
    with pytest.raises(PermissionError):
        data.load_real(allow_survivorship_bias=False)


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, allow_survivorship_bias=True, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(PANEL_CACHE), reason="offline CI: no real cache")
def test_real_panel_loads_from_cache():
    rets, caps = data.load_real(fetch=False, allow_survivorship_bias=True)
    assert rets.shape[1] == len(caps)
    assert list(rets.columns) == list(caps.index)
    assert (caps > 0).all()
