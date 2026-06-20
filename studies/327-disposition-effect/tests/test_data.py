"""The synthetic panel is well-formed and deterministic; the overhang construction is sane;
the real loader is cache-safe and refuses to run survivorship-biased without the opt-in."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disposition_effect import data  # noqa: E402

# Cache-gating: any test that reads a real/shared cache parquet must skip when it is absent.
_STUDY_CACHE = data._panel_cache_path(data.STUDY_CACHE)
_SHARED_CACHE = data._panel_cache_path(data.SHARED_CACHE)
_HAVE_CACHE = os.path.exists(_STUDY_CACHE) or os.path.exists(_SHARED_CACHE)


# ---- synthetic panel shape & determinism ----------------------------------
def test_synthetic_shape(planted):
    over, mom, fwd, truth = planted
    assert over.shape == fwd.shape == mom.shape
    assert list(over.columns) == list(fwd.columns)
    assert truth.has_overhang_edge
    # decorative monthly index must be a real DatetimeIndex, well under the ns overflow span.
    assert isinstance(over.index, pd.DatetimeIndex)
    assert over.index.year.max() < 2200


def test_synthetic_is_deterministic():
    a, _, _, _ = data.synthetic_panel(seed=5)
    b, _, _, _ = data.synthetic_panel(seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    c, _, _, _ = data.synthetic_panel(seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy(), equal_nan=True)


def test_overhang_has_both_signs(planted):
    """Some firms are deep in the money (g>0), some underwater (g<0)."""
    over, _, _, _ = planted
    vals = over.to_numpy()
    vals = vals[np.isfinite(vals)]
    assert (vals > 0).any() and (vals < 0).any()


def test_overhang_correlates_with_momentum(planted):
    """Mechanical check: a high-overhang stock is also a recent winner. The disposition
    'premium' is therefore confounded with momentum — the study's central caveat."""
    over, mom, _, _ = planted
    common = over.index.intersection(mom.index)
    o = over.loc[common].to_numpy().ravel()
    m = mom.loc[common].to_numpy().ravel()
    ok = np.isfinite(o) & np.isfinite(m)
    r = np.corrcoef(o[ok], m[ok])[0, 1]
    assert r > 0.2  # clearly positively correlated


# ---- Grinblatt-Han reference price ----------------------------------------
def test_reference_price_between_min_and_max():
    """A turnover-weighted average of past prices must lie within their range."""
    rng = np.random.default_rng(0)
    n = 1400
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                   index=pd.date_range("2010-01-01", periods=n, freq="B"))
    vol = pd.Series(rng.integers(1e6, 5e6, n).astype(float), index=px.index)
    ref = data.reference_price(px, vol, window=1260).dropna()
    for t in ref.index:
        loc = px.index.get_loc(t)
        win = px.iloc[loc - 1260:loc]
        assert win.min() - 1e-6 <= ref.loc[t] <= win.max() + 1e-6


def test_overhang_sign_tracks_trend():
    """On a steadily rising tape the overhang is positive (price above cost basis);
    on a falling tape it is negative."""
    n = 1400
    up = pd.Series(100 * np.exp(np.linspace(0, 0.5, n)),
                   index=pd.date_range("2010-01-01", periods=n, freq="B"))
    down = pd.Series(100 * np.exp(np.linspace(0, -0.5, n)), index=up.index)
    vol = pd.Series(2e6, index=up.index)
    g_up = data.overhang_from_bars(up, vol).dropna()
    g_dn = data.overhang_from_bars(down, vol).dropna()
    assert g_up.mean() > 0
    assert g_dn.mean() < 0


# ---- real loader: survivorship opt-in & cache-safety -----------------------
def test_load_real_requires_optin():
    with pytest.raises(data.SurvivorshipBiasError):
        data.load_real(allow_survivorship_bias=False)


def test_load_real_cache_miss_returns_empty(tmp_path):
    over, mom, fwd = data.load_real(
        allow_survivorship_bias=True, fetch=False, cache_dir=str(tmp_path)
    )
    assert over.empty and mom.empty and fwd.empty


@pytest.mark.skipif(not _HAVE_CACHE, reason="offline CI: no real-tape cache parquet")
def test_load_real_cache_is_well_formed():
    over, mom, fwd = data.load_real(allow_survivorship_bias=True, fetch=False)
    assert not over.empty
    assert isinstance(over.index, pd.DatetimeIndex)
    assert list(over.columns) == list(fwd.columns)


# ---- fingerprint ----------------------------------------------------------
def test_fingerprint_stable_and_content_sensitive(planted):
    over, _, _, _ = planted
    assert data.fingerprint(over) == data.fingerprint(over)
    other, _, _, _ = data.synthetic_panel(seed=99)
    assert data.fingerprint(over) != data.fingerprint(other)
