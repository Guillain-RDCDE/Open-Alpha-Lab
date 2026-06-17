"""Tests for the data layer of Study 300 (Sports-Sentiment).

All tests are offline and deterministic -- the hardcoded elimination table and
the synthetic generator only. No network, no parquet cache required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sports_sentiment import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded elimination table
# ---------------------------------------------------------------------------
def test_table_has_events(elim_table):
    assert len(elim_table) == 60


def test_table_columns(elim_table):
    expected = {"event_id", "country", "etf", "tournament", "match_date",
                "stage", "shootout", "note"}
    assert expected.issubset(set(elim_table.columns))


def test_event_ids_sequential(elim_table):
    assert list(elim_table["event_id"]) == list(range(1, 61))


def test_match_dates_parse_and_in_window(elim_table):
    import pandas as pd
    d = pd.to_datetime(elim_table["match_date"])
    assert d.min().year >= 1996
    assert d.max().year <= 2025


def test_tournaments_valid(elim_table):
    assert set(elim_table["tournament"].unique()).issubset({"WC", "EURO", "COPA"})


def test_shootout_is_bool(elim_table):
    assert set(elim_table["shootout"].unique()).issubset({True, False})


def test_etfs_nonempty_strings(elim_table):
    assert elim_table["etf"].str.len().gt(0).all()


def test_table_is_copy():
    t1 = data.elimination_table()
    t2 = data.elimination_table()
    t1.loc[0, "country"] = "XXX"
    assert t2.loc[0, "country"] != "XXX"


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_events"]
    assert set(df.columns) >= {"event_id", "next_day_ret"}


def test_synthetic_deterministic():
    a, _ = data.synthetic_panel(n_events=50, loss_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(n_events=50, loss_bps=0.0, seed=42)
    assert np.allclose(a["next_day_ret"], b["next_day_ret"])


def test_synthetic_seeds_differ():
    a, _ = data.synthetic_panel(n_events=50, loss_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(n_events=50, loss_bps=0.0, seed=2)
    assert not np.allclose(a["next_day_ret"], b["next_day_ret"])


def test_synthetic_null_mean_near_zero():
    df, _ = data.synthetic_panel(n_events=5000, loss_bps=0.0, seed=300)
    # base drift is ~3 bps; mean should be small (< 10 bps)
    assert abs(df["next_day_ret"].mean() * 1e4) < 10.0


def test_synthetic_loss_shifts_mean_down():
    df, _ = data.synthetic_panel(n_events=5000, loss_bps=49.0, seed=300)
    # mean should be roughly base(3) - 49 ~ -46 bps, clearly negative
    assert df["next_day_ret"].mean() * 1e4 < -30.0


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(n_events=40, loss_bps=0.0, seed=10)
    b, _ = data.synthetic_panel(n_events=40, loss_bps=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache-only behaviour: offline by default
# ---------------------------------------------------------------------------
def test_fetch_raises_without_cache(tmp_path):
    """With no parquet cache and fetch=False, fetch_event_returns raises."""
    with pytest.raises(FileNotFoundError):
        data.fetch_event_returns(cache_dir=str(tmp_path), fetch=False)
