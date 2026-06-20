"""The synthetic panel is well-formed and deterministic; the real loader is cache-safe
and survivorship-guarded."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fifty_two_week_range import data  # noqa: E402


def test_synthetic_shape_and_positive(null_panel):
    panel, truth = null_panel
    assert panel.shape == (truth["n_days"], truth["n_stocks"])
    assert (panel.to_numpy() > 0).all()
    assert list(panel.columns) == [f"S{i:02d}" for i in range(truth["n_stocks"])]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_days=400, low_edge=0.01, seed=7)
    b, _ = data.synthetic_panel(n_days=400, low_edge=0.01, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_panel(n_days=400, low_edge=0.01, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_null_panel_is_near_martingale(null_panel):
    """With no planted edge, day-to-day log returns carry no strong autocorrelation."""
    panel, _ = null_panel
    r = np.diff(np.log(panel.to_numpy()), axis=0)
    # Mean lag-1 autocorrelation across stocks should be tiny.
    acs = [np.corrcoef(r[:-1, j], r[1:, j])[0, 1] for j in range(r.shape[1])]
    assert abs(np.mean(acs)) < 0.05


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    # An empty cache_dir AND a bogus ticker (not in shared 202 cache) must raise.
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOTATICKER", fetch=False, cache_dir=str(tmp_path))


def test_load_panel_requires_survivorship_optin():
    with pytest.raises(PermissionError):
        data.load_panel(allow_survivorship_bias=False)


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    panel, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_panel(n_days=1500, range_edge=0.0, low_edge=0.0, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


# --- cache-gated: only runs when the shared real tape is present (offline-safe CI) ---
_SHARED_AAPL = data._cache_path("AAPL", data.SHARED_CACHE)


@pytest.mark.skipif(not os.path.exists(_SHARED_AAPL), reason="offline CI: no real cache")
def test_real_cache_loads_when_present():
    panels = data.load_ohlc_panels(allow_survivorship_bias=True)
    assert set(panels) == {"close", "high", "low"}
    assert panels["close"].shape[1] == len(data.SP500_BASKET)
    # high >= close >= low everywhere (basic OHLC sanity on the real tape).
    assert (panels["high"].to_numpy() >= panels["low"].to_numpy() - 1e-6).all()
