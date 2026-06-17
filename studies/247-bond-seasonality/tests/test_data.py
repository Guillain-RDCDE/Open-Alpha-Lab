"""Data-layer invariants for Study 247 (Bond-Seasonality).

All tests run OFFLINE on the synthetic tape.  The real-tape test is cache-gated.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_seasonality import data  # noqa: E402

_CACHE_TLT = os.path.join(
    os.path.dirname(__file__), "..", "_cache", "TLT.parquet"
)


def test_shape_and_columns(planted_tape):
    frame, truth = planted_tape
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"]
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(planted=True, seed=7)
    b, _ = data.synthetic_daily(planted=True, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(planted=True, seed=1)
    b, _ = data.synthetic_daily(planted=True, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_planted_truth_records_bump():
    _, truth = data.synthetic_daily(planted=True, bump=0.004, seed=247)
    assert truth["bump"] == 0.004
    _, truth0 = data.synthetic_daily(planted=False, seed=247)
    assert truth0["bump"] == 0.0


def test_tom_mask_covers_known_days():
    """TOM mask covers last-2 and first-2 trading days of each month."""
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    mask = data.tom_mask(idx, n_end=2, n_start=2)
    periods = idx.to_period("M")
    for p in periods.unique():
        days = idx[periods == p].sort_values()
        for d in days[:2]:
            assert mask.loc[d], f"Expected TOM on {d} (first-of-month)"
        for d in days[-2:]:
            assert mask.loc[d], f"Expected TOM on {d} (last-of-month)"


def test_tom_mask_excludes_mid_month():
    """Mid-month days are NOT flagged as TOM."""
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    mask = data.tom_mask(idx, n_end=2, n_start=2)
    periods = idx.to_period("M")
    for p in periods.unique():
        days = idx[periods == p].sort_values()
        if len(days) > 6:
            mid = days[3:-3]
            for d in mid:
                assert not mask.loc[d], f"Expected non-TOM on {d}"


def test_tom_mask_aligns_to_index(planted_tape):
    frame, truth = planted_tape
    mask = data.tom_mask(frame.index)
    assert len(mask) == len(frame)
    assert mask.dtype == bool
    assert mask.sum() == truth["n_tom"]


@pytest.mark.skipif(not os.path.exists(_CACHE_TLT), reason="real-tape cache absent offline/CI")
def test_load_real_shape():
    """Real tape returns a single 'close' column with positive values."""
    frame = data.load_real("TLT", start="2002-07-30")
    assert "close" in frame.columns
    assert len(frame) > 5000
    assert (frame["close"] > 0).all()
