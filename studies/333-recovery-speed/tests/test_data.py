"""The synthetic panel is well-formed and deterministic; the real loader is cache-safe
and survivorship-guarded."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery_speed import data  # noqa: E402

CACHE = data.DEFAULT_CACHE
_MKT_CACHE = data._cache_path(data.MARKET_PROXY, CACHE)


def test_synthetic_shape(null_panel):
    panel, market, truth = null_panel
    assert panel.shape == (truth["n_days"], truth["n_names"])
    assert market.shape[0] == truth["n_days"]
    assert (panel.to_numpy() > 0).all()
    assert list(panel.index) == list(market.index)


def test_synthetic_is_deterministic():
    a, ma, _ = data.synthetic_panel(n_names=30, n_days=400, recovery_edge=0.0005, seed=7)
    b, mb, _ = data.synthetic_panel(n_names=30, n_days=400, recovery_edge=0.0005, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ma.to_numpy(), mb.to_numpy())
    c, _, _ = data.synthetic_panel(n_names=30, n_days=400, recovery_edge=0.0005, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_planted_drawdown_is_deep_enough():
    """The engineered market drawdown actually reaches the planted depth (roughly)."""
    _, market, truth = data.synthetic_panel(n_names=20, n_days=1500, seed=333)
    px = market.to_numpy()
    run_max = np.maximum.accumulate(px)
    dd = float(-(px / run_max - 1.0).min())
    # The realised trough should be a meaningful fraction of the planted depth.
    assert dd > 0.6 * truth["drawdown_depth"]


def test_recovery_edge_couples_q_to_post_return():
    """With recovery_edge>0 the per-name latent q predicts post-recovery return; at 0 it
    is uncorrelated (the null)."""
    for edge, expect_positive in [(0.0, False), (0.0015, True)]:
        panel, _, truth = data.synthetic_panel(
            n_names=120, n_days=1500, recovery_edge=edge, seed=333
        )
        post = truth["post_idx"]
        fwd = (panel.iloc[-1] / panel.iloc[post] - 1.0).to_numpy()
        corr = np.corrcoef(truth["q"], fwd)[0, 1]
        if expect_positive:
            assert corr > 0.25
        else:
            assert abs(corr) < 0.20


def test_load_real_requires_survivorship_optin(tmp_path):
    with pytest.raises(PermissionError):
        data.load_real(fetch=False, cache_dir=str(tmp_path),
                       allow_survivorship_bias=False)


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path),
                       allow_survivorship_bias=True)


def test_fingerprint_stable_and_content_sensitive(null_panel):
    panel, _, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _, _ = data.synthetic_panel(n_names=80, n_days=1500, recovery_edge=0.0, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


@pytest.mark.skipif(not os.path.exists(_MKT_CACHE), reason="offline CI — no real cache")
def test_real_panel_loads_when_cached():
    panel, market, meta = data.load_real(allow_survivorship_bias=True)
    assert panel.shape[1] >= 1
    assert len(market) == len(panel)
    assert "survivorship_note" in meta and meta["n_names"] >= 1
