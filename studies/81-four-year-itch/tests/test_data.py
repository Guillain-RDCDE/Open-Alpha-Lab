"""The synthetic tape is well-formed, deterministic, and correctly labelled; cache is safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from four_year_itch import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert "ret" in df.columns
    assert "term_year" in df.columns
    assert len(df) == truth["n_days"]
    assert (df["term_year"].dropna().isin([1, 2, 3, 4])).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_cycles=10, year_premia=(0.0, 0.0, 0.0, 0.0), seed=42)
    b, _ = data.synthetic_daily(n_cycles=10, year_premia=(0.0, 0.0, 0.0, 0.0), seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(n_cycles=10, year_premia=(0.0, 0.0, 0.0, 0.0), seed=99)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_term_year_labels_cover_all_four_values(null_tape):
    df, _ = null_tape
    present = set(df["term_year"].dropna().astype(int).unique())
    assert present == {1, 2, 3, 4}


def test_term_year_label_consistency():
    """Each calendar year gets exactly one dominant term-year label."""
    import pandas as pd
    idx = pd.bdate_range("1929-01-01", periods=252 * 20, name="date")
    ty = data.term_year_label(idx)
    # Within a calendar year, labels should not mix (they are assigned by calendar year)
    for yr, grp in ty.dropna().groupby(idx[ty.notna()].year):
        uniq = grp.unique()
        assert len(uniq) == 1, f"year {yr} got mixed labels {uniq}"


def test_planted_premium_raises_year3_mean(signal_tape, null_tape):
    """The year_premia knob really shifts the planted year-of-term mean."""
    from four_year_itch import strategy as st
    ann_sig = st.annual_returns(signal_tape[0])
    ann_null = st.annual_returns(null_tape[0])
    y3_sig = ann_sig.loc[ann_sig["term_year"] == 3, "annual_ret"].mean()
    y3_null = ann_null.loc[ann_null["term_year"] == 3, "annual_ret"].mean()
    assert y3_sig > y3_null + 0.01  # planted +15 bps/day ≈ +3.8% annual lift


def test_fetch_prices_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_prices("NOPE", cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_cycles=10, year_premia=(0.0, 0.0, 0.0, 0.0), seed=7)
    assert data.fingerprint(df) != data.fingerprint(other)
