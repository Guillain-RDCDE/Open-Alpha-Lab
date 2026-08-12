"""Offline tests for the data layer — Study 845 (Stadium Naming-Rights Curse).

Deal table integrity, the tradable/untradable split, the fingerprint, and the
deterministic synthetic generator with its tunable curse knob. All synthetic-only; the
one real-cache probe is skipped when no cache is present (offline CI safe).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stadium_curse import data  # noqa: E402

CACHE = data.SPY_CACHE


# --------------------------------------------------------------------------- #
# The deal table
# --------------------------------------------------------------------------- #
def test_deal_table_not_empty():
    assert len(data.DEALS) >= 20


def test_deal_table_columns():
    df = data.deal_table()
    assert {"date", "venue", "sponsor", "ticker", "tradable", "note"}.issubset(df.columns)


def test_deal_table_chronological():
    d = data.deal_table()["date"]
    assert (d.diff().dropna() >= pd.Timedelta(0)).all()


def test_enron_present_and_untradable():
    """Enron Field — the origin myth — must be in the table AND flagged untradable."""
    df = data.deal_table()
    enron = df[df["sponsor"].str.contains("Enron")]
    assert len(enron) == 1
    assert bool(enron["tradable"].iloc[0]) is False


def test_ftx_present_and_untradable():
    df = data.deal_table()
    ftx = df[df["sponsor"].str.contains("FTX")]
    assert len(ftx) == 1
    assert bool(ftx["tradable"].iloc[0]) is False


def test_tradable_subset_all_flagged():
    tr = data.tradable_deals()
    assert tr["tradable"].all()
    assert len(tr) >= 20


def test_untradable_have_no_or_dead_ticker():
    """The cautionary tales are private or delisted — none is in the tradable set."""
    df = data.deal_table()
    untradable = df[~df["tradable"]]
    assert len(untradable) >= 3
    assert set(untradable["ticker"]).isdisjoint(set(data.tradable_tickers()))


def test_tradable_tickers_unique_sorted():
    ts = data.tradable_tickers()
    assert ts == sorted(set(ts))


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_stable_and_sensitive():
    idx = pd.bdate_range("2020-01-01", periods=100)
    s1 = pd.Series(np.linspace(100, 120, 100), index=idx)
    s2 = pd.Series(np.linspace(100, 120, 100), index=idx)
    s3 = pd.Series(np.linspace(100, 121, 100), index=idx)
    assert data.fingerprint(s1) == data.fingerprint(s2)
    assert data.fingerprint(s1) != data.fingerprint(s3)
    assert len(data.fingerprint(s1)) == 12


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
def test_synthetic_shapes(null_world):
    spy, prices, events = null_world
    assert len(prices) == len(events) == 28
    assert len(spy) == 8000
    for t, s in prices.items():
        assert len(s) == 8000


def test_synthetic_deterministic():
    a = data.synthetic_world(edge=-0.25, seed=845)
    b = data.synthetic_world(edge=-0.25, seed=845)
    for t in a[1]:
        assert np.allclose(a[1][t].to_numpy(), b[1][t].to_numpy())
    c = data.synthetic_world(edge=-0.25, seed=999)
    # a different seed changes the tapes
    some_ticker = next(iter(a[1]))
    assert not np.allclose(a[1][some_ticker].to_numpy(), c[1][some_ticker].to_numpy())


def test_synthetic_index_no_overflow(null_world):
    """The synthetic index must be a valid (non-overflowing) daily DatetimeIndex."""
    spy, _, _ = null_world
    assert isinstance(spy.index, pd.DatetimeIndex)
    assert spy.index.max().year < 2100


def test_curse_knob_lowers_sponsor_returns():
    """edge < 0 must pull sponsor tapes below their null-world counterparts on average."""
    _, pr_null, _ = data.synthetic_world(edge=0.0, seed=845)
    _, pr_curse, _ = data.synthetic_world(edge=-0.30, seed=845)
    end_null = np.mean([s.iloc[-1] for s in pr_null.values()])
    end_curse = np.mean([s.iloc[-1] for s in pr_curse.values()])
    assert end_curse < end_null


# --------------------------------------------------------------------------- #
# Real-cache probe — skipped when no cache present (offline CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_and_covers_spy():
    spy, prices = data.load_prices()
    assert len(spy) > 3000
    assert len(prices) >= 12
    assert spy.index.max() <= pd.Timestamp(data.AS_OF)
