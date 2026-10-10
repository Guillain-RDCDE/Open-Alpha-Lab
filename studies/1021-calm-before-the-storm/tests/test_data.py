"""Data-layer tests for Study 1021 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes absent — synthetic tests cover the logic")
needs_ff = pytest.mark.skipif(not data.have_ff(), reason="arch Fama-French tape absent")


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_monthly(n_months=400, seed=1021)
    b, _ = data.synthetic_monthly(n_months=400, seed=1021)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_monthly(n_months=400, seed=1021)
    b, _ = data.synthetic_monthly(n_months=400, seed=1022)
    assert not np.allclose(a["mkt"].to_numpy(), b["mkt"].to_numpy())


def test_synthetic_shape_index_and_horizon(planted):
    f, truth = planted
    assert len(f) == truth["n_months"]
    assert isinstance(f.index, pd.DatetimeIndex) and f.index.is_monotonic_increasing
    assert f.index[-1] < pd.Timestamp("2262-01-01")
    assert (f["mkt"] > -1).all()
    assert np.allclose(f["mkt"], f["mkt_rf"] + f["rf"])


def test_signal_strength_scales_the_plant():
    _, t1 = data.synthetic_monthly(n_months=300, signal_strength=1.0)
    _, th = data.synthetic_monthly(n_months=300, signal_strength=0.5)
    _, t0 = data.synthetic_monthly(n_months=300, signal_strength=0.0)
    assert t0["calm_crash_slope_eff"] == 0.0
    assert t1["calm_crash_slope_eff"] > th["calm_crash_slope_eff"] > 0.0


def test_the_null_is_matched_on_crash_count(planted, null_world):
    """The null changes WHEN crashes happen, not how many — else it would be a strawman."""
    n1 = planted[1]["n_crashes"]
    n0 = null_world[1]["n_crashes"]
    assert 0.6 < n1 / n0 < 1.6


def test_crash_months_are_bad_months(planted):
    f, truth = planted
    start = f.index[f["crash_start"]]
    nxt = [f.index.get_loc(d) + 1 for d in start if f.index.get_loc(d) + 1 < len(f)]
    assert f["mkt"].iloc[nxt].mean() < -0.04


def test_online_calm_matches_the_strategy_definition(planted):
    """The plant sees exactly the signal the strategy computes — no hidden mismatch."""
    f, _ = planted
    c = st.calm_signal(f["mkt"])["calm"]
    ok = c.notna()
    assert ok.sum() > 1000
    assert np.allclose(c[ok], f.loc[ok, "calm_true"], atol=1e-10)


def test_synthetic_volatility_clusters(null_world):
    f, _ = null_world
    a = np.abs(f["mkt_rf"].to_numpy())
    assert np.corrcoef(a[1:], a[:-1])[0, 1] > 0.1


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_monthly(n_months=300, seed=1)
    b, _ = data.synthetic_monthly(n_months=300, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_as_of_dates_are_month_ends_in_the_past():
    for d in (data.AS_OF, data.AS_OF_FF, data.AS_OF_SPX, data.AS_OF_VIX):
        ts = pd.Timestamp(d)
        assert ts.is_month_end and ts < pd.Timestamp("2026-01-01")


def test_monthly_rv_from_daily_on_a_known_series():
    idx = pd.bdate_range("2000-01-03", periods=60)
    r = np.where(np.arange(60) % 2 == 0, 0.01, -0.01)
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    rv = data.monthly_rv_from_daily(px)
    assert np.allclose(rv.iloc[1:-1], 0.01 * np.sqrt(252), rtol=1e-6)


# --------------------------------------------------------------------------- #
# Real tapes
# --------------------------------------------------------------------------- #
@needs_ff
def test_ff_is_total_return_and_pinned():
    ff = data.load_ff()
    assert ff.index[0] == pd.Timestamp("1926-07-31")
    assert ff.index[-1] == pd.Timestamp(data.AS_OF_FF)
    assert np.allclose(ff["mkt"], ff["mkt_rf"] + ff["rf"])
    assert len(ff) == 1109
    assert ff["rf"].between(-0.01, 0.02).all()


@needs_real
def test_spx_and_vix_drop_partial_months():
    spx = data.load_spx_daily()
    vix = data.load_vix()
    assert spx.index[-1] <= pd.Timestamp(data.AS_OF_SPX)
    assert spx.index[-1] >= pd.Timestamp("2022-11-25")
    assert vix.index[-1] <= pd.Timestamp(data.AS_OF_VIX)
    assert (spx > 0).all() and (vix > 5).all()


@needs_real
def test_provenance_names_every_tape_and_its_label():
    rows = data.provenance()
    assert len(rows) == 3
    labels = " ".join(r["label"] for r in rows)
    assert "total return" in labels and "price only" in labels
    assert all(len(r["sha256"]) == 64 for r in rows)
