"""Data-layer tests — synthetic determinism, the hardcoded event calendar, and a
skipif-guarded real-cache smoke test. Everything here runs offline with no cache."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methodology_shock import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_daily(seed=919)
    b, eb, _ = data.synthetic_daily(seed=919)
    assert np.allclose(a["treated"].to_numpy(), b["treated"].to_numpy())
    assert np.allclose(a["control"].to_numpy(), b["control"].to_numpy())
    assert [e["announce"] for e in ea] == [e["announce"] for e in eb]


def test_synthetic_seed_changes_the_tape():
    a, _, _ = data.synthetic_daily(seed=919)
    b, _, _ = data.synthetic_daily(seed=920)
    assert not np.allclose(a["treated"].to_numpy(), b["treated"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, events, truth = data.synthetic_daily(n_years=15, n_events=6, seed=919)
    assert {"treated", "control", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 15 * data.TRADING_DAYS_PER_YEAR
    assert len(events) == 6
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _, _ = data.synthetic_daily(seed=919)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_plants_nothing():
    _, _, t1 = data.synthetic_daily(signal_strength=1.0, seed=919)
    _, _, t0 = data.synthetic_daily(signal_strength=0.0, seed=919)
    assert t1["planted_car_bps"] > 0
    assert t0["planted_car_bps"] == 0.0


def test_synthetic_panel_pairs_are_independent():
    pairs, events, truth = data.synthetic_panel(n_pairs=3, n_events_per_pair=4, seed=919)
    assert set(pairs) == {"pair0", "pair1", "pair2"}
    assert len(events) == 12 == truth["n_events"]
    assert all("pair" in e for e in events)
    a = pairs["pair0"]["treated"].to_numpy()
    b = pairs["pair1"]["treated"].to_numpy()
    assert not np.allclose(a, b)


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_daily(seed=919)
    b, _, _ = data.synthetic_daily(seed=920)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


# --------------------------------------------------------------------------- #
# The hardcoded event calendar (a PROXY input — it must at least be coherent)
# --------------------------------------------------------------------------- #
def test_event_calendar_is_well_formed():
    for e in data.EVENTS:
        assert e["treated"] in data.TICKERS and e["control"] in data.TICKERS
        assert e["treated"] != e["control"]
        assert e["date_confidence"] in ("exact", "approximate")
        assert data.leg_date(e, "effective") is not None
    ids = [e["event_id"] for e in data.EVENTS]
    assert len(ids) == len(set(ids))


def test_effective_never_precedes_announcement():
    for e in data.EVENTS:
        a = data.leg_date(e, "announce")
        if a is not None:
            assert a <= data.leg_date(e, "effective")


def test_shared_announcement_is_not_double_counted():
    """Float-adjustment phase 2 shared phase 1's press release: no second announce leg."""
    ph2 = next(e for e in data.EVENTS if e["event_id"] == "spx-float-adjustment-phase-2")
    assert data.leg_date(ph2, "announce") is None
    ann = data.usable_events(leg="announce")
    eff = data.usable_events(leg="effective")
    assert ph2 not in ann and ph2 in eff


def test_usable_events_never_slices_the_future():
    for leg in ("announce", "effective"):
        for e in data.usable_events(leg=leg):
            assert data.leg_date(e, leg) <= pd.Timestamp(data.AS_OF)
    # The semi-annual reconstitution goes live in Nov 2026, past the as-of.
    ids = [e["event_id"] for e in data.usable_events(leg="effective")]
    assert "russell-semiannual-reconstitution" not in ids


def test_events_frame_parses_dates():
    df = data.events_frame()
    assert len(df) == len(data.EVENTS)
    assert pd.api.types.is_datetime64_any_dtype(df["effective"])


# --------------------------------------------------------------------------- #
# The offline contract
# --------------------------------------------------------------------------- #
def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (fresh checkout / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_event_study_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    s = st.run_event_study(px, list(data.EVENTS), leg="announce",
                           window=(1, 10), asof=data.AS_OF)
    assert s["n_events"] >= 5
    assert np.isfinite(s["mean_car_bps"]) and np.isfinite(s["t_cross_event"])
    # Betas must be sane for large, liquid, same-country equity wrappers.
    assert s["table"]["beta"].between(0.2, 2.0).all()
