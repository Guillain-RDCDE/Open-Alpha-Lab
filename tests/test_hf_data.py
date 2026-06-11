"""Tests for the free HuggingFace single-name reader (quantlab.hf_data).

The deterministic tests (guard + adjustment maths) run everywhere with just pandas.
The live push-down test needs duckdb + network, so it skips when either is absent —
matching the desk convention that real-data tests are cache/network-gated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import hf_data as hf


def test_panel_guard_raises_without_optin():
    """A cross-sectional panel must REFUSE to build unless the bias is opted into.

    The guard fires before any network call, so this is fully deterministic.
    """
    with pytest.raises(hf.SurvivorshipBiasError):
        hf.load_panel(["AAPL", "MSFT", "IBM"])


def test_panel_optin_warns_and_delegates(monkeypatch):
    """With the opt-in, it warns and fetches each name (here stubbed — no network)."""
    calls = []

    def fake_load_symbol(sym, mode="total_return", use_cache=True):
        calls.append(sym)
        return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setattr(hf, "load_symbol", fake_load_symbol)
    with pytest.warns(UserWarning, match="survivorship"):
        panel = hf.load_panel(["AAPL", "MSFT"], allow_survivorship_bias=True)
    assert set(panel) == {"AAPL", "MSFT"}
    assert calls == ["AAPL", "MSFT"]


def test_apply_mode_total_return_vs_as_is():
    """total_return scales OHLC by adj_close/close; as_is leaves the reported close."""
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.0, 11.0],
            "volume": [100, 200],
            "adj_close": [5.0, 11.0],  # row 0 is half-adjusted (a 0.5 factor)
        }
    )
    as_is = hf._apply_mode(raw, "as_is")
    tr = hf._apply_mode(raw, "total_return")
    # as_is keeps the reported close untouched
    assert as_is["Close"].tolist() == [10.0, 11.0]
    # total_return applies the per-row adj_close/close factor (0.5 then 1.0)
    assert tr["Close"].tolist() == [5.0, 11.0]
    assert tr["Open"].tolist() == [5.0, 11.0]
    # columns match quantlab.data.fetch for drop-in use
    assert list(tr.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        hf.load_symbol("AAPL", mode="split_only")  # deliberately unsupported here


@pytest.mark.network
def test_live_single_name_pushdown():
    """Live: one real stock via DuckDB push-down. Skips without duckdb or network."""
    pytest.importorskip("duckdb")
    try:
        df = hf.load_symbol("AAPL", mode="total_return", use_cache=False)
    except Exception as exc:  # offline / endpoint down
        pytest.skip(f"HF endpoint unavailable: {type(exc).__name__}")
    assert len(df) > 5000  # AAPL has deep history (since 1980)
    assert df.index.min().year <= 1981
    assert (df["Close"] > 0).all()
    assert not np.isnan(df["Close"].iloc[-1])
