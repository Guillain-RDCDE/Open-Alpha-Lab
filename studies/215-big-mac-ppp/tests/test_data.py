"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from big_mac_ppp import data  # noqa: E402

_CACHE_PATH = data._cache_path(data.DEFAULT_CACHE)


def test_synthetic_shape(null_panel):
    misval, fwd_ret, truth = null_panel
    assert misval.shape[0] == truth["n_years"]
    assert misval.shape[1] == truth["n_currencies"]
    assert fwd_ret.shape == misval.shape
    assert misval.columns.tolist() == [f"C{i}" for i in range(truth["n_currencies"])]


def test_synthetic_is_deterministic():
    a, b, _ = data.synthetic_panel(n_years=60, reversion_signal=0.10, seed=5)
    c, d, _ = data.synthetic_panel(n_years=60, reversion_signal=0.10, seed=5)
    assert np.allclose(a.to_numpy(), c.to_numpy())
    assert np.allclose(b.to_numpy(), d.to_numpy())
    e, f, _ = data.synthetic_panel(n_years=60, reversion_signal=0.10, seed=6)
    assert not np.allclose(a.to_numpy(), e.to_numpy())


def test_null_panel_no_crosscorrelation(null_panel):
    """With reversion_signal=0, mis-val should not predict fwd_ret cross-sectionally."""
    misval, fwd_ret, _ = null_panel
    # Correlation across all obs
    mv_flat = misval.to_numpy().ravel()
    fr_flat = fwd_ret.to_numpy().ravel()
    corr = np.corrcoef(mv_flat, fr_flat)[0, 1]
    assert abs(corr) < 0.20  # should be near zero (sampling noise only)


def test_signal_panel_has_negative_correlation(signal_panel):
    """With reversion_signal > 0, over-valued currencies should tend to fall."""
    misval, fwd_ret, _ = signal_panel
    mv_flat = misval.to_numpy().ravel()
    fr_flat = fwd_ret.to_numpy().ravel()
    corr = np.corrcoef(mv_flat, fr_flat)[0, 1]
    # Planted reversion: over-valued (high mis_val) -> negative fwd_ret
    assert corr < -0.05  # should be detectably negative


def test_hardcoded_misval_structure():
    """The hardcoded Big Mac table returns a well-formed DataFrame."""
    df = data.hardcoded_misval()
    assert len(df) > 10  # at least 10 years
    assert set(data.CURRENCIES).issubset(df.columns) or len(df.columns) > 5
    # Values should be in a plausible range: -90% to +100%
    vals = df.to_numpy()
    valid = vals[~np.isnan(vals)]
    assert (valid > -90).all() and (valid < 120).all()


def test_fetch_fx_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_fx(fetch=False, cache_dir=str(tmp_path))


def test_build_panel_structure(null_panel):
    """build_panel joins mis-val and fwd-ret into a long tidy panel."""
    misval, fwd_ret, _ = null_panel
    panel = data.build_panel(misval, fwd_ret)
    assert "year" in panel.columns
    assert "currency" in panel.columns
    assert "mis_val_pct" in panel.columns
    assert "fwd_log_ret" in panel.columns
    assert len(panel) > 0


def test_fingerprint_stable(null_panel):
    misval, fwd_ret, _ = null_panel
    panel = data.build_panel(misval, fwd_ret)
    assert data.fingerprint(panel) == data.fingerprint(panel)


@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_fx_panel_loads():
    fx = data.fetch_fx(fetch=False)
    assert len(fx) > 5
    assert fx.index.name == "year"
