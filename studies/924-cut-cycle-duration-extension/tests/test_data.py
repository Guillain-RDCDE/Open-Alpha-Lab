"""Data-layer tests — the hardcoded calendar, synthetic determinism, and cache hygiene.

Everything here is offline. The only test that touches the real cache is guarded by a
``skipif`` on ``have_real``, so a fresh checkout with no ``_cache/`` is green.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_cut import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hardcoded FOMC calendar (an ASSUMPTION — so its shape is at least tested)
# --------------------------------------------------------------------------- #
def test_first_cuts_are_sorted_unique_dates():
    ev = data.first_cuts()
    assert len(ev) == 5
    assert ev.is_monotonic_increasing
    assert ev.is_unique
    assert isinstance(ev, pd.DatetimeIndex)


def test_all_cuts_contain_every_first_cut():
    first = set(data.first_cuts())
    allc = set(data.all_cuts())
    assert first.issubset(allc)
    assert len(allc) > len(first)


def test_all_cuts_sorted_and_truncated_for_full_windows():
    ev = data.all_cuts()
    assert ev.is_monotonic_increasing and ev.is_unique
    # Every listed cut must leave room for a full 12-month window inside the as-of.
    limit = pd.Timestamp(data.AS_OF) - pd.DateOffset(months=12)
    assert ev[-1] <= limit


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_daily(seed=924)
    b, eb, _ = data.synthetic_daily(seed=924)
    assert np.allclose(a["duration"].to_numpy(), b["duration"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())
    assert list(ea) == list(eb)


def test_synthetic_seed_changes_the_tape():
    a, _, _ = data.synthetic_daily(seed=924)
    b, _, _ = data.synthetic_daily(seed=925)
    assert not np.allclose(a["duration"].to_numpy(), b["duration"].to_numpy())


def test_synthetic_shape_columns_and_events():
    prices, events, truth = data.synthetic_daily(n_years=20, n_events=5, seed=924)
    assert {"duration", "front", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 20 * data.TRADING_DAYS_PER_YEAR
    assert len(events) == 5
    assert events.isin(prices.index).all()
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _, _ = data.synthetic_daily(seed=924)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_kills_the_planted_effect():
    _, _, t1 = data.synthetic_daily(signal_strength=1.0, seed=924)
    _, _, t0 = data.synthetic_daily(signal_strength=0.0, seed=924)
    assert t1["planted_6m_excess"] > 0.05
    assert t0["planted_6m_excess"] == 0.0


def test_synthetic_panel_worlds_are_distinct_and_sized():
    worlds = data.synthetic_panel(n_worlds=4, signal_strength=0.0, base_seed=924)
    assert len(worlds) == 4
    firsts = [w[0]["duration"].iloc[100] for w in worlds]
    assert len(set(np.round(firsts, 8))) == 4


# --------------------------------------------------------------------------- #
# Cash proxy & fingerprint
# --------------------------------------------------------------------------- #
def test_cash_index_from_irx_is_monotone():
    idx = pd.bdate_range("2010-01-04", periods=500)
    irx = pd.Series(np.linspace(0.05, 5.0, 500), index=idx)
    cash = data.cash_index_from_irx(irx)
    assert (cash.diff().dropna() > 0).all()
    assert cash.iloc[-1] > 1.0


def test_cash_index_from_irx_clips_negative_quotes():
    idx = pd.bdate_range("2010-01-04", periods=50)
    irx = pd.Series(-0.1, index=idx)
    cash = data.cash_index_from_irx(irx)
    assert np.allclose(cash.to_numpy(), 1.0)


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_daily(seed=924)
    b, _, _ = data.synthetic_daily(seed=925)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_event_study_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    dur, cash = px["TLT"].dropna(), px["BIL"].dropna()
    common = dur.index.intersection(cash.index)
    cmp = st.compare(dur.loc[common], cash.loc[common], data.first_cuts(), horizon_months=12)
    assert cmp["n_events"] >= 3
    assert np.isfinite(cmp["mean_excess_pct"])
    assert np.isfinite(cmp["t_event_leg_hac"])
