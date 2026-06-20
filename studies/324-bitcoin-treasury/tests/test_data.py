"""The synthetic pair is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_treasury import data  # noqa: E402


def test_synthetic_shape(null_pair):
    frame, truth = null_pair
    assert len(frame) == truth["n_days"]
    assert list(frame.columns) == ["btc", "mstr"]
    assert (frame["btc"] > 0).all()
    assert (frame["mstr"] > 0).all()
    assert frame.index.tz is None


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_pair(n_days=400, alpha=0.0005, seed=7)
    b, _ = data.synthetic_pair(n_days=400, alpha=0.0005, seed=7)
    assert np.allclose(a[["btc", "mstr"]].to_numpy(), b[["btc", "mstr"]].to_numpy())
    c, _ = data.synthetic_pair(n_days=400, alpha=0.0005, seed=8)
    assert not np.allclose(a["mstr"].to_numpy(), c["mstr"].to_numpy())


def test_beta_is_recovered(null_pair):
    """The OLS slope of MSTR-on-BTC must recover the planted leverage."""
    from bitcoin_treasury import strategy as st
    frame, truth = null_pair
    dec = st.beta_decomposition(frame)
    assert abs(dec["beta"] - truth["beta"]) < 0.15  # ~1.8


def test_premium_adds_variance_not_drift(null_pair, premium_pair):
    """A swinging NAV premium fattens MSTR vol but does not lift its drift."""
    from bitcoin_treasury import strategy as st
    flat, _ = null_pair
    prem, _ = premium_pair
    r_flat = st.simple_returns(flat["mstr"]).std()
    r_prem = st.simple_returns(prem["mstr"]).std()
    assert r_prem > r_flat  # the premium wobble adds variance
    # Mean MSTR return is not systematically lifted by the premium wobble.
    m_flat = st.simple_returns(flat["mstr"]).mean()
    m_prem = st.simple_returns(prem["mstr"]).mean()
    assert abs(m_prem - m_flat) < 0.001


def test_load_real_falls_back_to_synthetic(tmp_path):
    frame, tape = data.load_real(cache_dir=str(tmp_path), n_days=300)
    assert tape == "synthetic"
    assert list(frame.columns) == ["btc", "mstr"]


def test_fetch_close_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_close("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_pair):
    frame, _ = null_pair
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_pair(n_days=2500, beta=1.8, alpha=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# --- cache-gated real-tape test (skips offline) ----------------------------
_REAL_CACHE = data._cache_path("MSTR", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_REAL_CACHE), reason="offline CI: no real cache")
def test_real_pair_loads_when_cached():
    frame = data.fetch_pair(fetch=False)
    assert list(frame.columns) == ["btc", "mstr"]
    assert len(frame) > 100
