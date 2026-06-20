"""The synthetic event panel is well-formed and deterministic; the real loader is
cache-safe (and cache-gated so CI stays offline)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lockup_expiry import data  # noqa: E402

CACHE = data.DEFAULT_CACHE
BENCH_CACHE = data._price_cache_path(data.BENCHMARK, CACHE)


def test_synthetic_shape(null_panel):
    panel, truth = null_panel
    assert list(panel.columns) == ["event", "tau", "abn_ret"]
    # one row per (event, tau) over [-window, +window]
    n_taus = 2 * truth["window"] + 1
    assert len(panel) == truth["n_events"] * n_taus
    assert panel["tau"].min() == -truth["window"]
    assert panel["tau"].max() == truth["window"]
    assert np.isfinite(panel["abn_ret"]).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_events=10, sag_bps=-100.0, seed=5)
    b, _ = data.synthetic_panel(n_events=10, sag_bps=-100.0, seed=5)
    assert np.allclose(a["abn_ret"].to_numpy(), b["abn_ret"].to_numpy())
    c, _ = data.synthetic_panel(n_events=10, sag_bps=-100.0, seed=6)
    assert not np.allclose(a["abn_ret"].to_numpy(), c["abn_ret"].to_numpy())


def test_planted_sag_sits_on_expiry_bar(sag_panel):
    """The mean abnormal return at tau=0 is clearly negative when a sag is planted,
    and ~0 at a far-from-event bar."""
    panel, _ = sag_panel
    at0 = panel[panel["tau"] == 0]["abn_ret"].mean()
    far = panel[panel["tau"] == -20]["abn_ret"].mean()
    assert at0 < -0.01            # the planted -300 bps shows up
    assert abs(far) < 0.01        # away from the event there is no sag


def test_null_has_no_sag(null_panel):
    """With no planted sag, the expiry bar's mean abnormal return is near zero."""
    panel, _ = null_panel
    at0 = panel[panel["tau"] == 0]["abn_ret"].mean()
    assert abs(at0) < 0.01


def test_basket_frame_well_formed():
    bf = data.basket_frame()
    assert list(bf.columns) == ["ticker", "listing"]
    assert len(bf) >= 25
    assert bf["listing"].is_monotonic_increasing


def test_fingerprint_stable_and_sensitive(null_panel):
    panel, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_panel(n_events=60, sag_bps=-50.0, seed=2)
    assert data.fingerprint(panel) != data.fingerprint(other)


@pytest.mark.skipif(not os.path.exists(BENCH_CACHE), reason="offline CI — no shared cache")
def test_real_panel_loads_from_cache():
    """When the cache exists, the real loader returns a well-formed long panel."""
    panel = data.fetch_event_panel(fetch=False)
    assert list(panel.columns) == ["event", "tau", "abn_ret"]
    assert len(panel) > 0
    assert panel["tau"].abs().max() <= data.WINDOW
