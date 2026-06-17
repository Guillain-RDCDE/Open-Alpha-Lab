"""The synthetic panel is well-formed, deterministic, and tunable — Study 229."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beneish_m_score import data  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY_ROOT = os.path.dirname(_HERE)
_SHARED_CACHE = os.path.abspath(os.path.join(_STUDY_ROOT, "..", "..", "_cache"))
_LOCAL_CACHE  = os.path.join(_STUDY_ROOT, "_cache")

_SHARED_OK = all(
    os.path.exists(os.path.join(_SHARED_CACHE, f"_edgar_{c}.parquet"))
    for c in data._SHARED_CONCEPTS
) and os.path.exists(os.path.join(_SHARED_CACHE, "_edgar_yrret.parquet"))

_LOCAL_OK = all(
    os.path.exists(os.path.join(_LOCAL_CACHE, f"_edgar_{c}.parquet"))
    for c in data._LOCAL_CONCEPTS
)

_FULL_CACHE_OK = _SHARED_OK and _LOCAL_OK


# ---------------------------------------------------------------------------
# Synthetic panel tests (offline, no cache dependency)
# ---------------------------------------------------------------------------

def test_synthetic_shape(has_premium):
    m, fwd, truth = has_premium
    assert m.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == m.shape


def test_synthetic_index_columns(has_premium):
    m, fwd, truth = has_premium
    assert m.index.name == "year"
    assert fwd.index.name == "year"
    assert list(m.index) == list(fwd.index)
    assert list(m.columns) == list(fwd.columns)


def test_synthetic_m_range(has_premium):
    """M-scores are clipped to a plausible range [-8, 4]."""
    m, fwd, truth = has_premium
    assert (m >= -8.0).all().all()
    assert (m <= 4.0).all().all()


def test_synthetic_is_deterministic():
    m1, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    m2, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    assert np.allclose(m1.values, m2.values)
    m3, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=99)
    assert not np.allclose(m1.values, m3.values)


def test_synthetic_null_knob(no_premium):
    """With m_premium=0 the truth dict correctly flags no premium."""
    _, _, truth = no_premium
    assert truth["m_premium"] == 0.0
    assert not truth["has_premium"]


def test_synthetic_premium_knob(has_premium):
    """With m_premium != 0 the truth dict flags a premium."""
    _, _, truth = has_premium
    assert truth["has_premium"]


def test_fingerprint_stable_and_content_sensitive(has_premium):
    m, _, _ = has_premium
    fp1 = data.fingerprint(m)
    fp2 = data.fingerprint(m)
    assert fp1 == fp2  # deterministic
    m2, _, _ = data.synthetic_panel(n_firms=200, n_years=20, seed=999)
    assert fp1 != data.fingerprint(m2)


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """fetch_panel with non-existent shared cache returns empty frames."""
    m, fwd = data.fetch_panel(
        shared_cache=str(tmp_path),
        local_cache=str(tmp_path),
        fetch_missing=False,
    )
    assert m.empty
    assert fwd.empty


@pytest.mark.skipif(not _FULL_CACHE_OK, reason="real-tape EDGAR cache absent offline/CI")
def test_fetch_panel_real_shape():
    """With all caches present the panel has expected shape."""
    m, fwd = data.fetch_panel(fetch_missing=False)
    assert not m.empty
    assert m.shape[0] >= 10   # at least 10 years
    assert m.shape[1] >= 20   # at least 20 tickers
