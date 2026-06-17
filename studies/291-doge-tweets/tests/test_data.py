"""Tests for the data layer of Study 291 (Doge-Tweets).

All tests are offline and deterministic — they use the hardcoded tweet table and
the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doge_tweets import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded tweet table
# ---------------------------------------------------------------------------
def test_table_nonempty(tweet_table):
    assert len(tweet_table) >= 20


def test_table_columns(tweet_table):
    assert {"date", "direction", "note"}.issubset(set(tweet_table.columns))


def test_table_dates_are_datetime(tweet_table):
    assert pd.api.types.is_datetime64_any_dtype(tweet_table["date"])


def test_table_dates_sorted_and_in_range(tweet_table):
    d = tweet_table["date"]
    assert d.min().year >= 2019
    assert d.max().year <= 2026
    assert list(d) == sorted(d)


def test_table_directions_valid(tweet_table):
    assert set(tweet_table["direction"].unique()).issubset({-1, 0, 1})


def test_table_mostly_bullish(tweet_table):
    """The curated famous list is dominated by bullish/promotional events."""
    assert (tweet_table["direction"] > 0).sum() >= 0.7 * len(tweet_table)


def test_table_known_event_present(tweet_table):
    """The 'Dogefather SNL' announcement (2021-04-28) is in the table."""
    assert (tweet_table["date"] == pd.Timestamp("2021-04-28")).any()


def test_table_is_copy():
    t1 = data.tweet_table()
    t2 = data.tweet_table()
    t1.loc[0, "direction"] = 99
    assert t2.loc[0, "direction"] != 99


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"doge_ret", "btc_ret", "is_event", "direction"}


def test_synthetic_event_count(synthetic_null):
    df, truth = synthetic_null
    assert int(df["is_event"].sum()) == truth["n_events"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=42)
    assert np.allclose(a["doge_ret"].to_numpy(), b["doge_ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=2)
    assert not np.allclose(a["doge_ret"].to_numpy(), b["doge_ret"].to_numpy())


def test_synthetic_null_no_event_bump():
    """On the null tape, event-day abnormal returns are not systematically large."""
    df, _ = data.synthetic_panel(n_days=2000, bump_bps=0.0, seed=291)
    ev = df.loc[df["is_event"], "doge_ret"].mean()
    noev = df.loc[~df["is_event"], "doge_ret"].mean()
    assert abs(ev - noev) < 0.05


def test_synthetic_signal_creates_event_bump():
    """A large planted bump raises mean DOGE return on event days vs off-event."""
    df, _ = data.synthetic_panel(n_days=2000, bump_bps=3000.0, seed=291)
    ev = df.loc[df["is_event"], "doge_ret"].mean()
    noev = df.loc[~df["is_event"], "doge_ret"].mean()
    assert ev > noev + 0.05


def test_prices_to_returns_shape():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {"DOGE-USD": np.linspace(1, 2, 10), "BTC-USD": np.linspace(100, 110, 10)},
        index=idx,
    )
    rets = data.prices_to_returns(prices)
    assert list(rets.columns) == ["doge_ret", "btc_ret"]
    assert len(rets) == 9  # first NaN row dropped


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=10)
    b, _ = data.synthetic_panel(n_days=300, bump_bps=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_prices_raises_without_cache(tmp_path):
    """Without the cache, fetch_prices(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_prices(fetch=False, cache_path=str(tmp_path / "missing.parquet"))
