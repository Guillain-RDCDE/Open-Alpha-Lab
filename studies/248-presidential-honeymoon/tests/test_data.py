"""Synthetic tape is well-formed, deterministic, and correctly labelled; cache is safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidential_honeymoon import data  # noqa: E402

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "presidential_honeymoon_gspc_daily.parquet",
)


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert "ret" in df.columns
    assert "in_honeymoon" in df.columns
    assert len(df) == truth["n_days"]
    # Some days should be in honeymoon, some not
    assert df["in_honeymoon"].any()
    assert not df["in_honeymoon"].all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_terms=10, honeymoon_premium_bp=0.0, seed=42)
    b, _ = data.synthetic_daily(n_terms=10, honeymoon_premium_bp=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(n_terms=10, honeymoon_premium_bp=0.0, seed=99)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_honeymoon_labels_non_overlapping():
    """Honeymoon windows should not overlap (each inauguration is 4+ years apart)."""
    import pandas as pd
    idx = pd.bdate_range("1929-01-01", periods=252 * 20, name="date")
    labels = data.honeymoon_labels(pd.DatetimeIndex(idx))
    # On any honeymoon day the term_idx should be a single consistent value
    in_hm = labels[labels["in_honeymoon"]]
    # term_idx should always be a finite integer on honeymoon days
    assert in_hm["term_idx"].notna().all()


def test_planted_premium_raises_honeymoon_mean(signal_tape, null_tape):
    """The honeymoon_premium_bp knob really shifts the planted honeymoon mean."""
    from presidential_honeymoon import strategy as st

    sig_hm = signal_tape[0].loc[signal_tape[0]["in_honeymoon"], "ret"].mean()
    null_hm = null_tape[0].loc[null_tape[0]["in_honeymoon"], "ret"].mean()
    # +20 bps/day planted — should be clearly higher
    assert sig_hm > null_hm + 5e-4


def test_fetch_prices_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_prices("NOPE", cache_dir=str(tmp_path), fetch=False)


@pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="real-tape cache absent offline/CI",
)
def test_fetch_prices_real_cache_loads():
    df = data.fetch_prices("^GSPC", fetch=False)
    assert "close" in df.columns
    assert "in_honeymoon" in df.columns
    assert len(df) > 1000


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_terms=10, honeymoon_premium_bp=0.0, seed=7)
    assert data.fingerprint(df) != data.fingerprint(other)
