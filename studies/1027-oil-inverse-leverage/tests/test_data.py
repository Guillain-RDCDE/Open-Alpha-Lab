"""Data-layer tests for Study 1027 — synthetic determinism offline, pinned real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mirrorlev import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes not installed — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_gjr(n_years=3, seed=1027)
    b, _ = data.synthetic_gjr(n_years=3, seed=1027)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_gjr(n_years=3, seed=1027)
    b, _ = data.synthetic_gjr(n_years=3, seed=1028)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_index_and_horizon():
    px, truth = data.synthetic_gjr(n_years=5, seed=1027)
    assert len(px) == 5 * data.TRADING_DAYS_PER_YEAR == truth["n_days"]
    assert isinstance(px.index, pd.DatetimeIndex) and px.index.is_monotonic_increasing
    assert px.index[-1] < pd.Timestamp("2262-01-01")
    assert (px > 0).all()


def test_signal_strength_sets_sign_and_size_of_gamma():
    t = {k: data.synthetic_gjr(n_years=1, signal_strength=k)[1] for k in (1.0, 0.5, 0.0, -1.0)}
    assert t[1.0]["gamma"] > t[0.5]["gamma"] > t[0.0]["gamma"] == 0.0 > t[-1.0]["gamma"]
    assert t[1.0]["gamma"] == pytest.approx(-t[-1.0]["gamma"])
    # the mirror swaps the two impacts exactly
    assert t[1.0]["down_impact"] == pytest.approx(t[-1.0]["up_impact"])
    assert t[1.0]["up_impact"] == pytest.approx(t[-1.0]["down_impact"])


def test_knob_leaves_persistence_and_unconditional_vol_alone():
    a = data.synthetic_gjr(n_years=1, signal_strength=1.0)[1]
    b = data.synthetic_gjr(n_years=1, signal_strength=-1.0)[1]
    c = data.synthetic_gjr(n_years=1, signal_strength=0.0)[1]
    assert a["persistence"] == b["persistence"] == c["persistence"] < 1.0
    assert a["omega"] == b["omega"] == c["omega"]


def test_synthetic_vol_is_near_target():
    px, truth = data.synthetic_gjr(n_years=20, signal_strength=0.0, seed=1027)
    vol = data.log_returns(px).std() * np.sqrt(252)
    assert vol == pytest.approx(truth["ann_vol"], rel=0.2)


def test_too_strong_a_knob_is_refused():
    with pytest.raises(ValueError):
        data.synthetic_gjr(n_years=1, signal_strength=2.0)


def test_returns_helpers():
    px = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2000-01-03", periods=3), name="x")
    assert data.simple_returns(px).iloc[0] == pytest.approx(0.10)
    assert data.log_returns(px).iloc[0] == pytest.approx(np.log(1.1))
    assert len(data.log_returns(px)) == 2


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_gjr(n_years=1, seed=1)
    b, _ = data.synthetic_gjr(n_years=1, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_as_of_is_a_month_end_in_the_past():
    for d in (data.AS_OF, data.AS_OF_MONTHLY):
        ts = pd.Timestamp(d)
        assert ts == ts + pd.offsets.MonthEnd(0)
        assert ts < pd.Timestamp.today()


# --------------------------------------------------------------------------- #
# Real tapes (frozen in the arch wheel, SHA-256 pinned)
# --------------------------------------------------------------------------- #
@needs_real
def test_wti_spot_is_pinned_and_clean():
    w = data.load_wti()
    assert w.index[0] == pd.Timestamp("1986-01-02")
    assert w.index[-1] <= pd.Timestamp(data.AS_OF)
    assert w.index.is_monotonic_increasing and not w.index.has_duplicates
    assert (w > 0).all() and not w.isna().any()
    assert len(w) > 8000


@needs_real
def test_partial_january_2019_is_dropped():
    assert data.load_wti().index[-1].year == 2018


@needs_real
def test_sp500_is_the_price_index():
    s = data.load_sp500()
    assert s.index[0] == pd.Timestamp("1999-01-04") and s.index[-1] == pd.Timestamp("2018-12-31")
    assert s.between(500, 3500).all()


@needs_real
def test_crude_monthly_is_an_average_of_the_daily_spot():
    c = data.load_crude_monthly()
    w = data.load_wti().resample("ME").mean()
    both = pd.concat([c["wti"], w], axis=1, join="inner").dropna()
    assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().median() < 0.05
    assert c.index[-1] == pd.Timestamp(data.AS_OF_MONTHLY)


@needs_real
def test_rf_is_nan_beyond_the_french_tape_never_filled():
    s = data.load_sp500()
    rf = data.load_rf_daily(s.index)
    assert rf.loc["2018-12"].isna().all()
    assert rf.loc["2010"].notna().all() and (rf.loc["2010"] >= 0).all()


@needs_real
def test_provenance_carries_the_sha_pins():
    rows = data.provenance()
    assert {r["tape"] for r in rows} == set(data.TAPES)
    assert all(len(r["sha256"]) == 64 and len(r["fingerprint"]) == 12 for r in rows)
