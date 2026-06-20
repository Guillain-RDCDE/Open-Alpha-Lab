"""The synthetic tape is well-formed and deterministic; the crack identity is correct;
the real loader is cache-safe (and cache-gated when it would read a shared parquet)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_spread import data  # noqa: E402


def test_synthetic_shape_and_columns(coincident):
    frame, truth = coincident
    assert len(frame) == truth["n_days"]
    assert list(frame.columns) == ["crack", "stock", "stock_ret"]
    assert (frame["stock"] > 0).all()
    assert frame["crack"].notna().all()
    # The decorative index must not overflow ns-timestamps (well under ~290 years).
    assert frame.index[-1].year < 2100


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_tape(n_days=500, lead=0.2, seed=7)
    b, _ = data.synthetic_tape(n_days=500, lead=0.2, seed=7)
    assert np.allclose(a["crack"].to_numpy(), b["crack"].to_numpy())
    assert np.allclose(a["stock_ret"].to_numpy(), b["stock_ret"].to_numpy())
    c, _ = data.synthetic_tape(n_days=500, lead=0.2, seed=8)
    assert not np.allclose(a["crack"].to_numpy(), c["crack"].to_numpy())


def test_lead_creates_predictive_correlation(coincident, leading):
    """The lead knob really induces lag-1 predictability (and lead=0 induces ~none)."""
    def lag1_corr(f):
        d = f["crack"].diff().shift(1)
        joint = pd.concat([d, f["stock_ret"]], axis=1).dropna()
        return np.corrcoef(joint.iloc[:, 0], joint.iloc[:, 1])[0, 1]

    null, _ = coincident
    lead, _ = leading
    assert abs(lag1_corr(null)) < 0.06          # null: no lag-1 predictability
    assert lag1_corr(lead) > 0.15               # planted lead is detectable


def test_crack_is_contemporaneous_even_in_the_null(coincident):
    """Even with lead=0 the crack co-moves with the stock *same day* — the whole point:
    coincident is not predictive."""
    null, _ = coincident
    d0 = null["crack"].diff()
    joint = pd.concat([d0, null["stock_ret"]], axis=1).dropna()
    same_day = np.corrcoef(joint.iloc[:, 0], joint.iloc[:, 1])[0, 1]
    assert same_day > 0.3                        # strong same-day co-movement


def test_crack_321_identity():
    """The 3-2-1 crack matches the closed-form: (2*RB*42 + HO*42)/3 - CL."""
    rb = pd.Series([2.0, 2.5])
    ho = pd.Series([2.2, 2.3])
    cl = pd.Series([80.0, 90.0])
    got = data.crack_321(rb, ho, cl)
    exp = (2.0 * rb * 42.0 + ho * 42.0) / 3.0 - cl
    assert np.allclose(got.to_numpy(), exp.to_numpy())


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coincident):
    frame, _ = coincident
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_tape(n_days=4000, lead=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# --- cache-gated: only runs when the shared real parquet actually exists offline ---
_CRACK_CACHE = data._cache_path("CL=F", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_CRACK_CACHE), reason="offline CI (no real cache)")
def test_load_real_from_cache_has_expected_shape():
    frame = data.load_real(fetch=False)
    assert list(frame.columns) == ["crack", "stock", "stock_ret"]
    assert len(frame) > 1000
    assert (frame["stock"] > 0).all()
