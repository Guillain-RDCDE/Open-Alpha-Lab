"""Tests for the data layer of Study 158 (Super-Bowl).

All tests are offline and deterministic — they use the hardcoded Super Bowl
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from super_bowl import data  # noqa: E402


# ---------------------------------------------------------------------------
# Super Bowl hardcoded table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(sb_table):
    """We have 59 Super Bowls (I through LIX, 1967–2025)."""
    assert len(sb_table) == 59


def test_table_columns(sb_table):
    expected = {"game_number", "game_year", "winner", "conference", "orig_nfl"}
    assert expected.issubset(set(sb_table.columns))


def test_table_game_numbers_sequential(sb_table):
    """Game numbers run from 1 to 59 with no gaps."""
    assert list(sb_table["game_number"]) == list(range(1, 60))


def test_table_game_years_cover_range(sb_table):
    """Games span 1967 through 2025 (all Super Bowl years)."""
    assert sb_table["game_year"].min() == 1967
    assert sb_table["game_year"].max() == 2025


def test_table_conferences_valid(sb_table):
    """Only 'NFC' and 'AFC' appear in the conference column."""
    assert set(sb_table["conference"].unique()) == {"NFC", "AFC"}


def test_table_orig_nfl_is_bool(sb_table):
    """orig_nfl column contains Python booleans."""
    assert sb_table["orig_nfl"].dtype == bool or set(sb_table["orig_nfl"].unique()).issubset({True, False})


def test_table_known_winners(sb_table):
    """Spot-check several well-known results."""
    # Super Bowl I: Green Bay Packers
    row1 = sb_table[sb_table["game_number"] == 1].iloc[0]
    assert "Green Bay" in row1["winner"]
    assert row1["conference"] == "NFC"
    assert bool(row1["orig_nfl"]) is True

    # Super Bowl III: NY Jets (AFL → AFC, not original NFL)
    row3 = sb_table[sb_table["game_number"] == 3].iloc[0]
    assert "Jets" in row3["winner"]
    assert row3["conference"] == "AFC"
    assert bool(row3["orig_nfl"]) is False

    # Super Bowl LIX (2025): Philadelphia Eagles (NFC)
    row59 = sb_table[sb_table["game_number"] == 59].iloc[0]
    assert "Eagles" in row59["winner"]
    assert row59["conference"] == "NFC"


def test_table_nfc_afc_split_is_reasonable(sb_table):
    """Both conferences should have between 15 and 45 wins each (rough check)."""
    nfc = (sb_table["conference"] == "NFC").sum()
    afc = (sb_table["conference"] == "AFC").sum()
    assert 15 <= nfc <= 45
    assert 15 <= afc <= 45
    assert nfc + afc == 59


def test_superbowl_table_is_copy(sb_table):
    """superbowl_table() returns a copy — mutating it does not affect the original."""
    t1 = data.superbowl_table()
    t2 = data.superbowl_table()
    t1.loc[0, "conference"] = "XXX"
    assert t2.loc[0, "conference"] != "XXX"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "return", "conference", "orig_nfl"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=42)
    b, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=42)
    assert np.allclose(a["return"].to_numpy(), b["return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=1)
    b, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=2)
    assert not np.allclose(a["return"].to_numpy(), b["return"].to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, NFC/AFC mean returns should be close (no planted effect)."""
    df, _ = data.synthetic_annual(n_years=1000, signal_bps=0.0, seed=158)
    nfc_mean = df.loc[df["conference"] == "NFC", "return"].mean()
    afc_mean = df.loc[df["conference"] == "AFC", "return"].mean()
    # With n=1000, the gap should be small (< 2%).
    assert abs(nfc_mean - afc_mean) < 0.02


def test_synthetic_signal_creates_difference():
    """A planted signal should create a measurable difference in NFC vs AFC returns."""
    df, _ = data.synthetic_annual(n_years=500, signal_bps=5000.0, seed=158)
    nfc_mean = df.loc[df["conference"] == "NFC", "return"].mean()
    afc_mean = df.loc[df["conference"] == "AFC", "return"].mean()
    assert nfc_mean > afc_mean + 0.3


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=10)
    df2, _ = data.synthetic_annual(n_years=30, signal_bps=0.0, seed=11)
    # Rename to look like sp500_return for fingerprint
    df1 = df1.rename(columns={"return": "sp500_return"})
    df2 = df2.rename(columns={"return": "sp500_return"})
    df1["mkt_up"] = df1["sp500_return"] > 0
    df2["mkt_up"] = df2["sp500_return"] > 0
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_sp500_annual_raises_without_cache(tmp_path):
    """Without the Shiller cache, fetch_sp500_annual raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500_annual(cache_dir=str(tmp_path))
