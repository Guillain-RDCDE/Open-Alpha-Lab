"""The curated call table is well-formed; the synthetic universe is deterministic and plants
what it claims; the real loader is cache-safe and gated for offline CI."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_cramer import data  # noqa: E402

# Where load_real reads its cache from (quantlab's shared parquet cache). Cache-dependent
# tests are skipped offline; synthetic tests never skip.
_QL_CACHE = os.environ.get(
    "OVERNIGHT_CACHE",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache")),
)
_STUDY_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


# ---- the curated call table -----------------------------------------------
def test_calls_table_shape_and_directions():
    df = data.calls_table()
    assert {"ticker", "date", "direction", "note"} <= set(df.columns)
    assert len(df) >= 15  # a real curated list, not a stub
    assert set(df["direction"].unique()) <= {-1, 1}
    assert df["date"].is_monotonic_increasing  # sorted by date
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---- synthetic universe shape & determinism --------------------------------
def test_synthetic_shape(null_world):
    prices, calls, truth = null_world
    assert len(prices) == truth["n_paths"]
    assert {"path", "call_idx", "direction", "fwd_ret"} <= set(calls.columns)
    assert len(calls) == truth["n_calls"]
    assert set(calls["direction"].unique()) <= {-1, 1}
    # every path is a tz-naive close series of the right length
    for s in prices.values():
        assert len(s) == truth["n_days"]
        assert getattr(s.index, "tz", None) is None
        assert (s > 0).all()


def test_synthetic_is_deterministic():
    _, a, _ = data.synthetic_calls(anti_alpha=0.3, selection_bias=0.0, seed=7)
    _, b, _ = data.synthetic_calls(anti_alpha=0.3, selection_bias=0.0, seed=7)
    assert np.allclose(a["fwd_ret"].to_numpy(), b["fwd_ret"].to_numpy())
    _, c, _ = data.synthetic_calls(anti_alpha=0.3, selection_bias=0.0, seed=8)
    assert not np.allclose(a["fwd_ret"].to_numpy(), c["fwd_ret"].to_numpy())


def test_anti_alpha_makes_calls_wrong(null_world, anti_world):
    """The anti_alpha knob makes direction*fwd_ret negative more often (pundit wrong)."""
    _, null_calls, _ = null_world
    _, anti_calls, _ = anti_world
    frac_wrong = lambda c: ((c["direction"] * c["fwd_ret"]) < 0).mean()
    # null ~ a coin (around half); anti clearly more often wrong
    assert abs(frac_wrong(null_calls) - 0.5) < 0.12
    assert frac_wrong(anti_calls) > frac_wrong(null_calls) + 0.15


def test_selection_bias_curates_against_pundit(curated_world):
    """With selection_bias the published list over-represents calls that went against him."""
    _, calls, _ = curated_world
    frac_wrong = ((calls["direction"] * calls["fwd_ret"]) < 0).mean()
    assert frac_wrong > 0.6  # curated to his worst calls


def test_fingerprint_stable_and_content_sensitive(null_world, anti_world):
    _, a, _ = null_world
    _, b, _ = anti_world
    assert data.fingerprint_calls(a) == data.fingerprint_calls(a)
    assert data.fingerprint_calls(a) != data.fingerprint_calls(b)


# ---- real loader: cache-safe, gated ----------------------------------------
def test_load_real_raises_without_cache(tmp_path, monkeypatch):
    """An empty cache (and no fetch) must raise, so callers can fall back to synthetic."""
    monkeypatch.setenv("OVERNIGHT_CACHE", str(tmp_path))
    # quantlab reads CACHE_DIR at import; force a fresh import under the empty cache.
    for m in list(sys.modules):
        if m == "quantlab.data" or m.startswith("quantlab.data."):
            del sys.modules[m]
    with pytest.raises((FileNotFoundError, Exception)):
        data.load_real(tickers=["NOPE_XYZ"], fetch=False)


@pytest.mark.skipif(not os.path.isdir(_QL_CACHE), reason="offline CI: no shared quantlab cache")
def test_load_real_returns_series_when_cached():
    """If any call ticker is cached, load_real returns clean tz-naive close series."""
    try:
        tapes = data.load_real(fetch=False)
    except FileNotFoundError:
        pytest.skip("offline CI: no call ticker cached")
    assert isinstance(tapes, dict) and len(tapes) >= 1
    for s in tapes.values():
        assert isinstance(s, pd.Series)
        assert getattr(s.index, "tz", None) is None
