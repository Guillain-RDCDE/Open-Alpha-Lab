"""Tests for the data layer of Study 294 (Coinbase-Rank).

All tests are offline and deterministic -- they use the hardcoded rank-spike
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinbase_rank import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded rank-spike table
# ---------------------------------------------------------------------------
def test_table_nonempty(spike_table):
    assert len(spike_table) >= 12


def test_table_columns(spike_table):
    assert {"date", "direction", "note"}.issubset(set(spike_table.columns))


def test_table_dates_are_datetime(spike_table):
    assert pd.api.types.is_datetime64_any_dtype(spike_table["date"])


def test_table_dates_sorted_and_in_range(spike_table):
    d = spike_table["date"]
    assert d.min().year >= 2017
    assert d.max().year <= 2026
    assert list(d) == sorted(d)


def test_table_directions_valid(spike_table):
    assert set(spike_table["direction"].unique()).issubset({-1, 0, 1})


def test_table_all_euphoria_spikes(spike_table):
    """Every curated spike is a +1 retail-euphoria 'toward #1' marker."""
    assert (spike_table["direction"] == 1).all()


def test_table_known_event_present(spike_table):
    """The Dec-2017 blow-off spike is in the table."""
    assert (spike_table["date"] == pd.Timestamp("2017-12-07")).any()


def test_table_is_copy():
    t1 = data.rank_spike_table()
    t2 = data.rank_spike_table()
    t1.loc[0, "direction"] = 99
    assert t2.loc[0, "direction"] != 99


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"btc_ret", "eth_ret", "is_event", "direction"}


def test_synthetic_event_count(synthetic_null):
    df, truth = synthetic_null
    assert int(df["is_event"].sum()) == truth["n_events"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=42)
    assert np.allclose(a["btc_ret"].to_numpy(), b["btc_ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=2)
    assert not np.allclose(a["btc_ret"].to_numpy(), b["btc_ret"].to_numpy())


def test_synthetic_null_no_post_spike_drift():
    """On the null tape, forward returns after events are not systematically negative."""
    df, _ = data.synthetic_panel(n_days=2500, bump_bps=0.0, seed=294)
    btc = df["btc_ret"].to_numpy()
    ev_pos = np.where(df["is_event"].to_numpy())[0]
    fwd = np.array([btc[p + 1:p + 6].sum() for p in ev_pos if p + 6 < len(btc)])
    assert abs(float(fwd.mean())) < 0.05


def test_synthetic_signal_creates_negative_drift():
    """A large planted bump makes forward returns after events systematically negative."""
    df, _ = data.synthetic_panel(n_days=2500, bump_bps=1500.0, seed=294)
    btc = df["btc_ret"].to_numpy()
    ev_pos = np.where(df["is_event"].to_numpy())[0]
    fwd = np.array([btc[p + 1:p + 6].sum() for p in ev_pos if p + 6 < len(btc)])
    assert float(fwd.mean()) < -0.05


def test_prices_to_returns_shape():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {"BTC-USD": np.linspace(100, 110, 10), "ETH-USD": np.linspace(10, 12, 10)},
        index=idx,
    )
    rets = data.prices_to_returns(prices)
    assert list(rets.columns) == ["btc_ret", "eth_ret"]
    assert len(rets) == 9  # first NaN row dropped


def test_fingerprint_stable_and_content_sensitive():
    a, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=10)
    b, _ = data.synthetic_panel(n_days=400, bump_bps=0.0, seed=11)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fetch_prices_raises_without_cache(tmp_path):
    """Without the cache, fetch_prices(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_prices(fetch=False, cache_path=str(tmp_path / "missing.parquet"))
