"""Reproducibility stamp — deterministic, offline, CI-friendly.

These tests guarantee the *property* the data stamp promises: the fingerprint is
stable for identical data and moves the instant the data does. If this suite
passes, a published fingerprint is a real, checkable claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import repro


def _frame(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"Close": 100 + rng.standard_normal(n).cumsum(), "Volume": rng.integers(1, 9, n)},
        index=idx,
    )


def test_fingerprint_is_deterministic():
    df = _frame()
    assert repro.fingerprint(df) == repro.fingerprint(df.copy())


def test_fingerprint_is_column_order_independent():
    df = _frame()
    assert repro.fingerprint(df) == repro.fingerprint(df[["Volume", "Close"]])


def test_fingerprint_changes_when_a_value_changes():
    df = _frame()
    bumped = df.copy()
    bumped.iloc[10, bumped.columns.get_loc("Close")] += 1e-3
    assert repro.fingerprint(df) != repro.fingerprint(bumped)


def test_fingerprint_changes_when_the_sample_grows():
    df = _frame(n=50)
    assert repro.fingerprint(df) != repro.fingerprint(_frame(n=51))


def test_fingerprint_is_stable_under_tiny_subround_noise():
    # Below the rounding precision -> same fingerprint (intended: ignore float dust).
    df = _frame()
    nudged = df.copy()
    nudged["Close"] += 1e-9
    assert repro.fingerprint(df) == repro.fingerprint(nudged)


def test_as_of_pins_the_sample():
    df = _frame(n=100)
    cut = df.index[40]
    sliced = repro.as_of(df, str(cut.date()))
    assert sliced.index.max() <= cut
    assert len(sliced) == 41
    assert len(repro.as_of(df, None)) == len(df)  # None as-of is a no-op


def test_data_stamp_mentions_rows_range_and_fingerprint():
    df = _frame(n=30)
    line = repro.data_stamp("TEST", df, asof="2000-12-31")
    assert "TEST" in line
    assert "30 rows" in line
    assert "as-of 2000-12-31" in line
    assert repro.fingerprint(df) in line
