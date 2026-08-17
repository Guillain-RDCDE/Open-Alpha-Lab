"""Data-layer tests — the hardcoded event table, synthetic determinism, cache discipline.

Every test here is offline and cache-free: the suite runs green on a fresh checkout where
``studies/_cache`` is absent. The one real-cache test is skipped cleanly in that case.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dutch_auction import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded EDGAR event table
# --------------------------------------------------------------------------- #
def test_event_table_shape_and_window():
    assert len(data.EVENTS) >= 150
    for tk, date, adsh, name in data.EVENTS:
        assert re.fullmatch(r"[A-Z][A-Z0-9.]{0,6}", tk), tk
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), date
        assert re.fullmatch(r"\d{10}-\d{2}-\d{6}", adsh), adsh
        assert name.strip()
        assert "2010-01-01" <= date <= "2025-12-31"


def test_events_are_sorted_and_have_no_exact_duplicates():
    dates = [d for _, d, _, _ in data.EVENTS]
    assert dates == sorted(dates)
    pairs = [(t, d) for t, d, _, _ in data.EVENTS]
    assert len(pairs) == len(set(pairs))


def test_repeat_offers_by_one_issuer_are_at_least_150_days_apart():
    """The clustering rule that turns filings into offers must leave no near-duplicates."""
    by_tk = {}
    for tk, d, _, _ in data.EVENTS:
        by_tk.setdefault(tk, []).append(pd.Timestamp(d))
    for tk, ds in by_tk.items():
        ds = sorted(ds)
        gaps = [(b - a).days for a, b in zip(ds, ds[1:])]
        assert all(g > 150 for g in gaps), (tk, gaps)


def test_ticker_helpers_exclude_recycled_symbols_and_add_benchmarks():
    ev = data.event_tickers()
    assert all(t not in ev for t in data.EXCLUDED_TICKERS)
    allt = data.all_tickers()
    assert data.MARKET in allt and data.CASH in allt
    assert set(ev).issubset(set(allt))


# --------------------------------------------------------------------------- #
# Synthetic generators (offline, deterministic)
# --------------------------------------------------------------------------- #
def test_synthetic_daily_is_deterministic_and_shaped():
    a = data.synthetic_daily(n_days=500, seed=927)
    b = data.synthetic_daily(n_days=500, seed=927)
    assert list(a.columns) == ["stock", "market", "cash"]
    assert len(a) == 500 and isinstance(a.index, pd.DatetimeIndex)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert a.index[-1] < pd.Timestamp("2262-01-01")
    assert (a["cash"].diff().dropna() > 0).all()


def test_synthetic_panel_is_deterministic_and_well_formed():
    p1, e1, t1 = data.synthetic_panel(n_events=20, n_days=800, seed=927)
    p2, e2, t2 = data.synthetic_panel(n_events=20, n_days=800, seed=927)
    assert set(p1) == set(p2) and e1 == e2 and t1 == t2
    assert "MKT" in p1 and "CASH" in p1
    assert len(e1) == 20 and t1["n_events"] == 20
    for name in p1:
        assert np.allclose(p1[name].to_numpy(), p2[name].to_numpy())
        assert (p1[name] > 0).all()
    for row in e1:
        assert len(row) == 4 and row[0] in p1


def test_signal_strength_zero_plants_nothing():
    _, _, t1 = data.synthetic_panel(n_events=10, n_days=800, signal_strength=1.0, seed=927)
    _, _, t0 = data.synthetic_panel(n_events=10, n_days=800, signal_strength=0.0, seed=927)
    assert t1["planted_jump"] > 0 and t1["planted_drift_6m"] > 0
    assert t0["planted_jump"] == 0.0 and t0["planted_drift_6m"] == 0.0


def test_fingerprint_stable_and_sensitive():
    a = data.synthetic_daily(n_days=400, seed=927)
    b = data.synthetic_daily(n_days=400, seed=928)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)
    d = {"X": a["stock"], "Y": a["market"]}
    assert len(data.fingerprint(d)) == 12
    assert data.fingerprint(d) == data.fingerprint({"Y": a["market"], "X": a["stock"]})


# --------------------------------------------------------------------------- #
# Cache discipline
# --------------------------------------------------------------------------- #
def test_study_fingerprint_ignores_history_outside_the_study_window():
    """The desk cache is shared, so a longer pull by a neighbouring study must not move it."""
    d = data.synthetic_daily(n_days=3000, start="2005-01-03", seed=927)
    full = {"A": d["stock"], "B": d["market"]}
    short = {k: v[v.index >= pd.Timestamp(data.FP_START)] for k, v in full.items()}
    assert 0 < len(short["A"]) < len(full["A"])     # the trim really removed rows
    fp = data.study_fingerprint(full)
    assert len(fp) == 12
    assert data.study_fingerprint(short) == fp


def test_study_fingerprint_survives_a_dividend_re_adjustment():
    """``auto_adjust=True`` rescales the WHOLE history when a new dividend is paid.

    Every return — and therefore every number in this study — is untouched by that, so the
    stamp must not move either. Hashing price *levels* would have flagged a phantom change,
    which is exactly the drift this desk saw twice on the shared cache.
    """
    d = data.synthetic_daily(n_days=2000, start="2008-01-02", seed=927)
    before = {"A": d["stock"], "B": d["market"]}
    after = {k: v * 1.00374 for k, v in before.items()}     # a fresh dividend, back-applied
    assert not np.allclose(after["A"].to_numpy(), before["A"].to_numpy())
    assert data.study_fingerprint(after) == data.study_fingerprint(before)


def test_study_fingerprint_moves_when_the_data_moves():
    a = data.synthetic_daily(n_days=3000, start="2005-01-03", seed=927)
    b = data.synthetic_daily(n_days=3000, start="2005-01-03", seed=928)
    assert data.study_fingerprint({"A": a["stock"]}) != data.study_fingerprint({"A": b["stock"]})


def test_load_prices_raises_without_the_market_leg(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_load_dollar_volume_is_empty_without_cache(tmp_path):
    assert data.load_dollar_volume(cache_dir=str(tmp_path)) == {}


def test_data_module_does_not_import_yfinance_at_module_scope():
    """``load_prices`` must be usable with no network stack present at all."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "dutch_auction", "data.py"), encoding="utf-8").read()
    top = [ln for ln in src.splitlines() if ln.startswith("import ") or ln.startswith("from ")]
    assert not any("yfinance" in ln for ln in top)


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (CI) — synthetic tests cover the logic")
def test_real_cache_event_study_runs():
    px = data.load_prices()
    assert max(s.index[-1] for s in px.values()) <= pd.Timestamp(data.AS_OF)
    panel = st.EventPanel(px, data.EVENTS, market=data.MARKET)
    tbl = st.event_table(panel)
    assert len(tbl) >= 50
    for c in ("ar0", "window", "m1", "m3", "m6"):
        assert np.isfinite(tbl[c]).all()
