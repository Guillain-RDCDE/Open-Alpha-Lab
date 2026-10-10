"""Data-layer tests for Study 1013 — the chronology, the synthetic world, the gated real tapes."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nberclock import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover "
                                       "the logic")


# --------------------------------------------------------------------------- #
# The hard-coded NBER chronology
# --------------------------------------------------------------------------- #
def test_sixteen_cycles_since_1926_in_order():
    c = data.cycles()
    assert len(c) == 16
    assert (c["trough"] > c["peak"]).all()
    assert c["peak"].is_monotonic_increasing
    assert (c["peak"].iloc[1:].to_numpy() > c["trough"].iloc[:-1].to_numpy()).all()
    assert c["peak"].iloc[0] == pd.Timestamp("1926-10-31")
    assert c["trough"].iloc[-1] == pd.Timestamp("2020-04-30")


def test_great_depression_is_43_months_and_covid_is_2():
    c = data.cycles().set_index("peak")
    assert c.loc[pd.Timestamp("1929-08-31"), "months"] == 43
    assert c.loc[pd.Timestamp("2020-02-29"), "months"] == 2


def test_twelve_announcements_six_of_each_kind():
    a = data.announcements()
    assert len(a) == 12
    assert (a["kind"] == "peak").sum() == 6 and (a["kind"] == "trough").sum() == 6
    assert a["date"].is_monotonic_increasing
    assert (a["date"] > a["reference"]).all()
    assert (a["lag_months"] > 0).all()


def test_announcements_match_the_chronology():
    c = data.cycles()
    a = data.announcements()
    refs_p = set(a.loc[a.kind == "peak", "reference"])
    refs_t = set(a.loc[a.kind == "trough", "reference"])
    assert refs_p <= set(c["peak"]) and refs_t <= set(c["trough"])


def test_known_announcement_lags():
    a = data.announcements().set_index("date")
    assert a.loc[pd.Timestamp("2008-12-01"), "lag_months"] == 12
    assert a.loc[pd.Timestamp("1992-12-22"), "lag_months"] == 21
    assert a.loc[pd.Timestamp("2020-06-08"), "lag_months"] == 4


# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #
def test_synthetic_world_is_deterministic():
    a = data.synthetic_world(n_years=40, n_cycles=6, seed=7)
    b = data.synthetic_world(n_years=40, n_cycles=6, seed=7)
    assert np.allclose(a["returns"].to_numpy(), b["returns"].to_numpy())
    assert a["announcements"].equals(b["announcements"])


def test_synthetic_world_seed_sensitive():
    a = data.synthetic_world(n_years=40, n_cycles=6, seed=7)
    b = data.synthetic_world(n_years=40, n_cycles=6, seed=8)
    assert not np.allclose(a["returns"]["mkt"].to_numpy(), b["returns"]["mkt"].to_numpy())


def test_synthetic_world_shape_and_horizon(planted):
    r = planted["returns"]
    assert set(r.columns) >= {"mkt", "rf", "mkt_rf"}
    assert len(r) == 92 * 12
    assert isinstance(r.index, pd.DatetimeIndex)
    assert r.index[-1] < pd.Timestamp("2262-01-01")
    assert (r["mkt"] > -1).all()
    assert planted["truth"]["n_announcements"] == 2 * planted["truth"]["n_cycles"]


def test_synthetic_calendar_is_well_formed(planted):
    c, a = planted["cycles"], planted["announcements"]
    assert (c["trough"] > c["peak"]).all()
    assert (a["date"] > a["reference"]).all()
    # announcement dates fall mid-month, exercising the date -> bar mapping
    assert (a["date"].dt.day == 15).all()


def test_signal_strength_scales_the_plant():
    t1 = data.synthetic_world(signal_strength=1.0, seed=3)["truth"]
    th = data.synthetic_world(signal_strength=0.5, seed=3)["truth"]
    t0 = data.synthetic_world(signal_strength=0.0, seed=3)["truth"]
    assert t0["announce_premium"] == 0.0 and t0["lead_months"] == 0 and t0["bear_depth"] == 0
    assert t1["announce_premium"] > th["announce_premium"] > 0
    assert t1["lead_months"] >= th["lead_months"] > 0


def test_null_world_has_no_abnormal_drift(null):
    mu = null["returns"]["mkt"].mean() * 12
    assert 0.0 < mu < 0.15


# --------------------------------------------------------------------------- #
# Fingerprint and provenance
# --------------------------------------------------------------------------- #
def test_fingerprint_stable_and_sensitive(planted, null):
    fp = data.fingerprint(planted["returns"])
    assert fp == data.fingerprint(planted["returns"]) and len(fp) == 12
    assert fp != data.fingerprint(null["returns"])


def test_provenance_names_pins_and_labels():
    p = data.provenance()
    assert len(p["monthly"]["sha256"]) == 64 and len(p["daily"]["sha256"]) == 64
    assert "TOTAL" in p["monthly"]["label"] and "PRICE" in p["daily"]["label"]


def test_as_of_dates_are_month_ends_and_not_future():
    for a in (data.AS_OF_MONTHLY, data.AS_OF_DAILY):
        t = pd.Timestamp(a)
        assert t == t + pd.offsets.MonthEnd(0)
        assert t < pd.Timestamp("2026-01-01")


def test_load_daily_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_daily(cache_dir=str(tmp_path))


def test_have_daily_false_on_empty_dir(tmp_path):
    assert data.have_daily(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_monthly_tape_is_total_return_and_pinned():
    m = data.load_monthly()
    assert m.index[0] == pd.Timestamp("1926-07-31")
    assert m.index[-1] == pd.Timestamp(data.AS_OF_MONTHLY)
    assert np.allclose(m["mkt"], m["mkt_rf"] + m["rf"])
    assert m["mkt"].abs().max() < 0.5          # decimal, not percent


@needs_real
def test_daily_tape_drops_the_partial_month():
    d = data.load_daily()
    assert d.index[0] == pd.Timestamp("1990-01-02")
    assert d.index[-1] <= pd.Timestamp(data.AS_OF_DAILY)
    assert d.index[-1].month == 11 and d.index[-1].year == 2022
    assert d.index.is_monotonic_increasing and not d.index.has_duplicates
    assert (d > 0).all()


@needs_real
def test_daily_to_monthly_takes_month_end_closes():
    d = data.load_daily()
    m = data.daily_to_monthly(d)
    assert m.loc["2020-03"].iloc[0] == d.loc["2020-03"].iloc[-1]
