"""Tests for the data layer of Study 277 (Trading-Cards).

All tests are offline and deterministic -- they use the hardcoded curated card
index and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_cards import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded curated card index
# ---------------------------------------------------------------------------
def test_table_columns(card_table):
    expected = {"year", "card_index", "card_return"}
    assert expected.issubset(set(card_table.columns))


def test_table_year_range(card_table):
    assert card_table["year"].min() == 1990
    assert card_table["year"].max() == 2025


def test_table_years_sequential(card_table):
    years = list(card_table["year"])
    assert years == list(range(1990, 2026))


def test_table_index_base_is_100(card_table):
    base = card_table.loc[card_table["year"] == 1990, "card_index"].iloc[0]
    assert base == 100.0


def test_table_index_positive(card_table):
    assert (card_table["card_index"] > 0).all()


def test_table_first_return_is_nan(card_table):
    """The first year has no prior level, so its return is NaN."""
    assert np.isnan(card_table["card_index"].pct_change().iloc[0])


def test_table_has_pandemic_boom(card_table):
    """The curated index must show a large 2020 jump (the documented mania)."""
    lvl = card_table.set_index("year")["card_index"]
    ret_2020 = lvl[2020] / lvl[2019] - 1
    assert ret_2020 > 0.5  # documented ~+90%


def test_table_has_2022_crash(card_table):
    """The curated index must show the 2022 drawdown."""
    lvl = card_table.set_index("year")["card_index"]
    ret_2022 = lvl[2022] / lvl[2021] - 1
    assert ret_2022 < -0.15


def test_table_2025_below_peak(card_table):
    """2025 level sits well below the 2021 peak (round trip)."""
    lvl = card_table.set_index("year")["card_index"]
    assert lvl[2025] < lvl[2021]


def test_card_index_table_is_copy():
    t1 = data.card_index_table()
    t2 = data.card_index_table()
    t1.loc[0, "card_index"] = -999.0
    assert t2.loc[0, "card_index"] != -999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "card_return", "gspc_return"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=42)
    assert np.allclose(a["card_return"].to_numpy(), b["card_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=2)
    assert not np.allclose(a["card_return"].to_numpy(), b["card_return"].to_numpy())


def test_synthetic_null_has_no_alpha():
    """On the null tape, card mean return ~ beta * equity mean (no structural alpha)."""
    df, truth = data.synthetic_index(n_years=4000, signal_bps=0.0, seed=277)
    resid = df["card_return"].mean() - truth["beta"] * df["gspc_return"].mean()
    assert abs(resid) < 0.01


def test_synthetic_signal_creates_alpha():
    """A planted signal lifts the card mean above its beta-implied level."""
    df, truth = data.synthetic_index(n_years=2000, signal_bps=2_000.0, seed=277)
    resid = df["card_return"].mean() - truth["beta"] * df["gspc_return"].mean()
    assert resid > 0.10  # ~0.20 planted


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_index(n_years=30, signal_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_gspc_annual_raises_without_cache(tmp_path):
    """Without the cache and fetch=False, fetch_gspc_annual raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_annual(cache_dir=str(tmp_path), fetch=False)
