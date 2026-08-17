"""Data-layer tests — the event table, synthetic determinism, and a skipif cache smoke test.

Everything here runs offline. The real-cache test is skipped entirely when the shared
``studies/_cache`` is absent, so a fresh CI checkout is green without a byte of network.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from when_issued import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded event table (a non-tape input — so it gets checked like one)
# --------------------------------------------------------------------------- #
def test_event_table_is_well_formed():
    assert len(data.SPINOFF_EVENTS) >= 20
    for ev in data.SPINOFF_EVENTS + data.DELISTED_EVENTS:
        assert set(ev) == {"parent", "child", "announce", "regular_way", "ratio", "name"}
        assert ev["parent"] != ev["child"]
        assert 0.0 < float(ev["ratio"]) <= 2.0


def test_announcement_always_precedes_regular_way():
    """A separation cannot be distributed before it is announced."""
    for ev in data.SPINOFF_EVENTS:
        ann = pd.Timestamp(ev["announce"])
        rw = pd.Timestamp(ev["regular_way"])
        assert ann < rw, ev["name"]
        # and the lead time is plausible for a US Form 10 process (2 months .. 4 years)
        assert 60 <= (rw - ann).days <= 1500, ev["name"]


def test_no_event_reaches_past_the_as_of():
    asof = pd.Timestamp(data.AS_OF)
    for ev in data.SPINOFF_EVENTS:
        assert pd.Timestamp(ev["regular_way"]) < asof


def test_child_tickers_are_unique():
    children = [e["child"] for e in data.SPINOFF_EVENTS]
    assert len(children) == len(set(children))


def test_delisted_events_are_disjoint_from_the_live_table():
    live = {e["child"] for e in data.SPINOFF_EVENTS}
    gone = {e["child"] for e in data.DELISTED_EVENTS}
    assert live.isdisjoint(gone)


def test_event_tickers_covers_benchmark_and_cash():
    tk = set(data.event_tickers())
    assert data.BENCHMARK in tk and data.CASH in tk
    for ev in data.SPINOFF_EVENTS:
        assert ev["parent"] in tk and ev["child"] in tk


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_daily(seed=930)
    b, eb, _ = data.synthetic_daily(seed=930)
    assert list(a.columns) == list(b.columns)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    assert ea == eb


def test_synthetic_seeds_differ():
    a, _, _ = data.synthetic_daily(seed=930)
    b, _, _ = data.synthetic_daily(seed=931)
    assert not np.allclose(np.nan_to_num(a.to_numpy()), np.nan_to_num(b.to_numpy()))


def test_synthetic_schema_matches_the_real_loader():
    prices, events, truth = data.synthetic_daily(n_events=8, seed=930)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert data.BENCHMARK in prices.columns and data.CASH in prices.columns
    assert truth["n_events"] == len(events) <= 8
    for ev in events:
        assert set(ev) == {"parent", "child", "announce", "regular_way", "ratio", "name"}
        assert ev["parent"] in prices.columns and ev["child"] in prices.columns
    # OOB-safe: the synthetic index stays well inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_child_has_no_price_before_regular_way():
    """The child cannot be priced before it exists — the loader must see NaN there."""
    prices, events, _ = data.synthetic_daily(n_events=5, seed=930)
    for ev in events:
        rw = pd.Timestamp(ev["regular_way"])
        before = prices[ev["child"]][prices.index < rw]
        assert before.isna().all()


def test_synthetic_cash_is_monotone_growing():
    prices, _, _ = data.synthetic_daily(seed=930)
    assert (prices[data.CASH].diff().dropna() > 0).all()


def test_signal_strength_scales_the_planted_effects():
    _, _, t1 = data.synthetic_daily(signal_strength=1.0, seed=930)
    _, _, t0 = data.synthetic_daily(signal_strength=0.0, seed=930)
    assert t1["planted_child_alpha_21d"] > 0
    assert t0["planted_child_alpha_21d"] == 0.0
    assert t0["planted_parent_runup_daily"] == 0.0


def test_synthetic_panel_wrapper_returns_an_alpha_panel():
    panel, truth = data.synthetic_panel(n_events=6, seed=930)
    assert len(panel) == truth["n_events"]
    for col in ("a_parent_net", "a_child_5_net", "a_child_21_net", "a_comb_net"):
        assert col in panel.columns


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_daily(n_events=5, seed=930)
    b, _, _ = data.synthetic_daily(n_events=5, seed=931)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_fingerprint_ignores_history_before_the_study_window():
    """The stamp identifies THIS study's data, not the shared cache's depth.

    Another study can re-fetch SPY or BIL with a different start date at any time. That
    must not move the stamp, or docs/results.md stops reproducing for a reason that has
    nothing to do with the numbers in it.
    """
    a, _, _ = data.synthetic_daily(n_events=5, seed=930)
    pad = pd.DataFrame(1.0, index=pd.bdate_range(end="2010-12-30", periods=500),
                       columns=a.columns)
    assert data.fingerprint(pd.concat([pad, a])) == data.fingerprint(a)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (offline / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_panel_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    panel = st.event_panel(px, data.SPINOFF_EVENTS)
    assert len(panel) >= 20
    assert (panel["rw0"] <= pd.Timestamp(data.AS_OF)).all()
    for col in ("a_parent_net", "a_child_5_net", "a_comb_net"):
        assert np.isfinite(panel[col].to_numpy(dtype=float)).all()
