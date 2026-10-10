"""Data-layer tests for Study 1017 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oldage import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover logic")


def test_synthetic_is_deterministic():
    a, ta = data.synthetic_monthly(n_months=300, seed=1017)
    b, tb = data.synthetic_monthly(n_months=300, seed=1017)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta == tb


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_monthly(n_months=300, seed=1017)
    b, _ = data.synthetic_monthly(n_months=300, seed=1018)
    assert not np.allclose(a["mkt"].to_numpy(), b["mkt"].to_numpy())


def test_synthetic_shape_index_and_columns():
    df, truth = data.synthetic_monthly(n_months=1109, seed=1017)
    assert len(df) == 1109 == truth["n_months"]
    assert list(df.columns) == ["mkt", "rf", "mkt_rf"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert (df.index == df.index + pd.offsets.MonthEnd(0)).all()     # month-ends
    assert df.index[-1] < pd.Timestamp("2262-01-01")                  # pandas ns horizon
    assert np.allclose(df["mkt_rf"], df["mkt"] - df["rf"])
    assert (df["mkt"] > -1).all()


def test_null_world_is_a_plain_random_walk():
    """signal_strength=0 never fires a planted crash, and equals the s=0 log-return draw."""
    df, truth = data.synthetic_monthly(n_months=1109, signal_strength=0.0, seed=1017)
    assert truth["n_crash"] == 0
    rng = np.random.default_rng(1017)
    eps = rng.normal(truth["mu"], truth["sigma"], 1109)
    assert np.allclose(np.log1p(df["mkt"].to_numpy()), eps)


def test_signal_strength_plants_crashes():
    _, t1 = data.synthetic_monthly(n_months=1109, signal_strength=1.0, seed=1017)
    _, th = data.synthetic_monthly(n_months=1109, signal_strength=0.3, seed=1017)
    _, t0 = data.synthetic_monthly(n_months=1109, signal_strength=0.0, seed=1017)
    assert t1["n_crash"] > 0 and t0["n_crash"] == 0
    assert t1["signal_strength"] > th["signal_strength"] > t0["signal_strength"]


def test_total_return_index_compounds():
    r = pd.Series([0.10, -0.10, 0.05])
    lv = data.total_return_index(r)
    assert lv.iloc[-1] == pytest.approx(1.1 * 0.9 * 1.05)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_monthly(n_months=200, seed=1017)
    b, _ = data.synthetic_monthly(n_months=200, seed=1018)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_as_of_dates_are_month_ends_and_not_future():
    for d in (data.AS_OF_FF, data.AS_OF_SP, data.AS_OF):
        ts = pd.Timestamp(d)
        assert ts == ts + pd.offsets.MonthEnd(0)
        assert ts < pd.Timestamp("2026-01-01")


def test_provenance_carries_pins_and_labels():
    prov = data.provenance()
    assert len(prov) == 2
    for p in prov:
        assert len(p["sha256"]) == 64
    assert "total return" in prov[0]["label"] and "price index" in prov[1]["label"]


@needs_real
def test_ff_tape_loads_total_return_and_is_pinned():
    ff = data.load_ff_monthly()
    assert ff.index[0] == pd.Timestamp("1926-07-31")
    assert ff.index[-1] == pd.Timestamp(data.AS_OF_FF)
    assert len(ff) == 1109
    assert np.allclose(ff["mkt"], ff["mkt_rf"] + ff["rf"])
    assert ff["mkt"].abs().max() < 0.5          # decimals, not percent


@needs_real
def test_sp_tape_drops_the_partial_month():
    sp = data.load_sp500_daily()
    assert sp.index[0] == pd.Timestamp("1990-01-02")
    assert sp.index[-1] <= pd.Timestamp(data.AS_OF_SP)
    assert sp.index[-1].month == 11 and sp.index[-1].year == 2022
    assert sp.index.is_monotonic_increasing and not sp.index.has_duplicates
    assert (sp > 0).all()
