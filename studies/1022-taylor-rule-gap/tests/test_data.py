"""Data-layer tests for Study 1022 — synthetic determinism offline, pinned real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from taylorgap import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch / statsmodels tapes not installed")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_quarterly(n_quarters=120, seed=1022)
    b, _ = data.synthetic_quarterly(n_quarters=120, seed=1022)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_quarterly(n_quarters=120, seed=1022)
    b, _ = data.synthetic_quarterly(n_quarters=120, seed=1023)
    assert not np.allclose(a["eq_xs"], b["eq_xs"])


def test_synthetic_shape_columns_and_horizon():
    p, t = data.synthetic_quarterly(n_quarters=150)
    assert len(p) == 150 == t["n_quarters"]
    assert {"tgap", "eq_xs", "mkt", "rf", "aaa", "d_aaa", "bond_ret", "bond_xs"} <= set(p.columns)
    assert isinstance(p.index, pd.DatetimeIndex) and p.index.is_monotonic_increasing
    assert p.index[-1] < pd.Timestamp("2262-01-01")
    assert np.allclose(p["mkt"] - p["rf"], p["eq_xs"])


def test_signal_strength_scales_the_plant_and_zero_is_null():
    _, t1 = data.synthetic_quarterly(signal_strength=1.0)
    _, th = data.synthetic_quarterly(signal_strength=0.5)
    _, t0 = data.synthetic_quarterly(signal_strength=0.0)
    assert t1["beta_eq"] < th["beta_eq"] < 0.0 == t0["beta_eq"]
    assert t1["beta_bond"] < 0.0 == t0["beta_bond"]


def test_null_keeps_the_gap_path():
    """Only the slope moves between worlds: the gap itself is identical."""
    a, _ = data.synthetic_quarterly(signal_strength=1.0)
    b, _ = data.synthetic_quarterly(signal_strength=0.0)
    assert np.allclose(a["tgap"], b["tgap"])


def test_synthetic_gap_has_the_planted_persistence():
    p, t = data.synthetic_quarterly(n_quarters=4000, rho=0.9)
    g = p["tgap"].to_numpy()
    rho = np.corrcoef(g[1:], g[:-1])[0, 1]
    assert rho == pytest.approx(0.9, abs=0.03)


def test_par_bond_duration_matches_closed_form_and_finite_difference():
    d, c = data.par_bond_duration_convexity(np.array([0.05]))
    assert d[0] == pytest.approx(12.55, abs=0.01)     # 20y par bond at 5%
    assert c[0] > 0


def test_long_bond_return_is_carry_when_yields_do_not_move():
    y = pd.Series([6.0, 6.0, 6.0])
    r = data.long_bond_return(y)
    assert r.iloc[1] == pytest.approx(0.06 / 4)


def test_long_bond_return_falls_when_yields_rise_and_is_convex():
    up = data.long_bond_return(pd.Series([6.0, 7.0])).iloc[1]
    dn = data.long_bond_return(pd.Series([6.0, 5.0])).iloc[1]
    assert up < 0 < dn
    assert dn - 0.015 > -(up - 0.015)                # convexity: gains exceed losses


def test_long_bond_approximation_tracks_exact_repricing():
    """Second-order expansion vs repricing the bond exactly, for a 100 bp move."""
    y0, y1, n = 0.06, 0.07, 40
    t = np.arange(1, n + 1)
    cf = np.full(n, 3.0)
    cf[-1] += 100
    exact = (cf / (1 + y1 / 2) ** t).sum() / 100 - 1 + y0 / 4
    approx = data.long_bond_return(pd.Series([6.0, 7.0])).iloc[1]
    assert approx == pytest.approx(exact, abs=0.005)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_quarterly(seed=1)
    b, _ = data.synthetic_quarterly(seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_as_of_is_not_in_the_future_and_follows_the_macro_tape():
    assert pd.Timestamp(data.AS_OF) <= pd.Timestamp("today")
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.MACRO_LAST)


@needs_real
def test_macrodata_is_pinned():
    assert data.macrodata_sha256() == data.MACRODATA_SHA256


@needs_real
def test_real_quarterly_tape_is_clean_and_ends_at_as_of():
    q = data.load_quarterly()
    assert q.index[0] == pd.Timestamp(data.START)
    assert q.index[-1] == pd.Timestamp(data.AS_OF)
    assert q.index.is_monotonic_increasing and not q.index.has_duplicates
    assert q.loc[:data.MACRO_LAST, ["realgdp", "cpi", "tbilrate", "unemp"]].notna().all().all()
    assert np.allclose(q["mkt"] - q["rf"], q["eq_xs"])
    assert q["aaa"].between(2, 20).all()
    assert q["rf"].between(0, 0.05).all()


@needs_real
def test_real_quarterly_returns_compound_three_months():
    ff = data.bundled.ff_monthly_total_return()
    q = data.load_equity_quarterly()
    m = ff.loc["1990-01-31":"1990-03-31", "mkt"]
    assert q.loc["1990-03-31", "mkt"] == pytest.approx((1 + m).prod() - 1)


@needs_real
def test_provenance_names_every_pin():
    pv = data.provenance()
    assert set(pv) >= {"statsmodels", "arch", "macrodata_sha256", "frenchdata_sha256",
                       "default_sha256"}
