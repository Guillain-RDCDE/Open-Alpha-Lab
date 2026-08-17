"""Data-layer tests — event-table hygiene and synthetic determinism, all offline.

Nothing here touches the network or the shared cache except the final smoke test, which
skips itself cleanly when ``studies/_cache`` is absent (a fresh CI checkout).
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odd_lot import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded event table (shared with Study 927)
# --------------------------------------------------------------------------- #
def test_events_shape_and_dates():
    assert len(data.EVENTS) >= 150
    for tk, date, accession, label in data.EVENTS:
        assert re.fullmatch(r"[A-Z][A-Z.\-]*", tk), tk
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), date
        assert re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession), accession
        assert isinstance(label, str) and label


def test_events_are_point_in_time_and_inside_the_asof():
    dates = pd.to_datetime([d for _, d, _, _ in data.EVENTS])
    assert dates.min() >= pd.Timestamp("2010-01-01")
    assert dates.max() <= pd.Timestamp(data.AS_OF)


def test_usable_events_drops_recycled_symbols_and_sorts():
    ue = data.usable_events()
    assert all(tk not in data.EXCLUDED_TICKERS for tk, _, _, _ in ue)
    dates = [d for _, d, _, _ in ue]
    assert dates == sorted(dates)


def test_ticker_helpers_agree():
    assert set(data.all_tickers()) == set(data.event_tickers()) | {data.MARKET, data.CASH}


def test_declared_assumptions_are_inside_their_grids():
    assert data.PREMIUM_DEFAULT in data.PREMIUM_GRID
    assert data.EXPIRY_DAYS_DEFAULT in (20, 21)
    assert data.PRORATION_DEFAULT in data.PRORATION_GRID
    assert data.TENDER_FEE_USD_DEFAULT in data.TENDER_FEE_GRID
    assert 1.00 in data.PRORATION_GRID  # the "under-subscribed, priority worthless" case


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #
def test_synthetic_daily_is_deterministic_and_shaped():
    a = data.synthetic_daily(seed=928)
    b = data.synthetic_daily(seed=928)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert {"stock", "market", "cash"} == set(a.columns)
    assert isinstance(a.index, pd.DatetimeIndex)
    assert a.index[-1] < pd.Timestamp("2262-01-01")   # OOB-safe on pandas 2.x


def test_synthetic_panel_is_deterministic():
    p1, e1, t1 = data.synthetic_panel(n_events=20, seed=928)
    p2, e2, t2 = data.synthetic_panel(n_events=20, seed=928)
    assert e1 == e2 and t1 == t2
    for k in p1:
        assert np.allclose(p1[k].to_numpy(), p2[k].to_numpy())


def test_synthetic_panel_seed_sensitive():
    p1, _, _ = data.synthetic_panel(n_events=20, seed=928)
    p2, _, _ = data.synthetic_panel(n_events=20, seed=929)
    assert not np.allclose(p1["SYN000"].to_numpy(), p2["SYN000"].to_numpy())


def test_synthetic_panel_has_market_and_cash_legs():
    panel, events, truth = data.synthetic_panel(n_events=12, seed=928)
    assert "MKT" in panel and "CASH" in panel
    assert len(events) == 12 == truth["n_events"]
    assert (panel["CASH"].diff().dropna() > 0).all()
    for name, date, accession, label in events:
        assert name in panel
        assert pd.Timestamp(date) in panel[name].index


def test_signal_strength_zero_plants_nothing():
    _, _, t0 = data.synthetic_panel(n_events=5, signal_strength=0.0, seed=928)
    _, _, t1 = data.synthetic_panel(n_events=5, signal_strength=1.0, seed=928)
    assert t0["runup_planted_bps"] == 0.0 and t0["giveback_planted_bps"] == 0.0
    assert t1["runup_planted_bps"] > 0.0 and t1["giveback_planted_bps"] > 0.0


def test_event_windows_have_room_on_both_sides():
    """Every planted event must leave room for the pre-anchor and the post-expiry mark."""
    panel, events, truth = data.synthetic_panel(n_events=40, seed=928)
    idx = panel["MKT"].index
    for name, date, _, _ in events:
        pos = int(idx.searchsorted(pd.Timestamp(date)))
        assert pos >= st.ANCHOR_LAG_DEFAULT
        assert pos + truth["expiry_days"] + st.POST_DAYS_DEFAULT < len(idx)


def test_fingerprint_stable_and_sensitive():
    p1, _, _ = data.synthetic_panel(n_events=8, seed=928)
    p2, _, _ = data.synthetic_panel(n_events=8, seed=929)
    fp = data.fingerprint(p1)
    assert fp == data.fingerprint(p1) and len(fp) == 12
    assert fp != data.fingerprint(p2)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_event_table_builds():
    px = data.load_prices()
    tbl = st.build_event_table(px, data.usable_events(), market_key=data.MARKET)
    assert len(tbl) > 100
    assert tbl["date"].max() <= pd.Timestamp(data.AS_OF)
    assert (tbl["entry_date"] > tbl["t0"]).all()          # the one execution lag
    assert (tbl["expiry_date"] > tbl["entry_date"]).all()
    assert (tbl["post_date"] > tbl["expiry_date"]).all()
