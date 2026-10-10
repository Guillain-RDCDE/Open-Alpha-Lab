"""Data-layer tests for Study 1025 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dontlook import data  # noqa: E402

needs_monthly = pytest.mark.skipif(not data.have_monthly(),
                                   reason="arch tapes not installed")
needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="skfolio tape not cached — synthetic tests cover logic")


def test_synthetic_monthly_is_deterministic_and_seed_sensitive():
    a, _ = data.synthetic_monthly(n_months=240, seed=1025)
    b, _ = data.synthetic_monthly(n_months=240, seed=1025)
    c, _ = data.synthetic_monthly(n_months=240, seed=1026)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_monthly_shape_and_index():
    df, truth = data.synthetic_monthly(n_months=1104)
    assert list(df.columns) == ["stock", "bond", "bill"]
    assert len(df) == 1104 == truth["n_months"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[-1] < pd.Timestamp("2262-01-01")
    assert (df > -1).all().all()


def test_signal_strength_scales_the_planted_premium():
    t1 = data.synthetic_monthly(n_months=12, signal_strength=1.0)[1]
    th = data.synthetic_monthly(n_months=12, signal_strength=0.5)[1]
    t0 = data.synthetic_monthly(n_months=12, signal_strength=0.0)[1]
    assert t0["equity_premium_monthly"] == 0.0
    assert t1["equity_premium_monthly"] == pytest.approx(2 * th["equity_premium_monthly"])


def test_planted_premium_has_the_size_it_claims():
    df, truth = data.synthetic_monthly(n_months=200000, signal_strength=1.0, seed=7)
    prem = (df["stock"] - df["bill"]).mean()
    assert prem == pytest.approx(truth["equity_premium_monthly"], abs=0.0005)
    df0, _ = data.synthetic_monthly(n_months=200000, signal_strength=0.0, seed=7)
    assert abs((df0["stock"] - df0["bill"]).mean()) < 0.0005


def test_synthetic_daily_shape():
    df, truth = data.synthetic_daily(n_years=3)
    assert len(df) == 3 * data.TRADING_DAYS_PER_YEAR == truth["n_days"]
    assert df.index[-1] < pd.Timestamp("2262-01-01")


def test_par_bond_duration_matches_textbook_values():
    D, C = data.par_bond_duration_convexity(np.array([0.06]), maturity=20.0)
    # semiannual 6% 20-year par bond: modified duration ≈ 11.56, convexity ≈ 187
    assert D[0] == pytest.approx(11.56, abs=0.05)
    assert C[0] == pytest.approx(187, rel=0.02)
    D5, _ = data.par_bond_duration_convexity(np.array([0.06]), maturity=5.0)
    assert D5[0] < D[0]


def test_duration_approximation_agrees_with_exact_repricing():
    rng = np.random.default_rng(1)
    y = pd.Series(6 + np.cumsum(rng.normal(0, 0.15, 600)),
                  index=pd.date_range("1950-01-31", periods=600, freq="ME"))
    a = data.bond_total_return(y, method="duration")
    b = data.bond_total_return(y, method="reprice")
    assert len(a) == 599
    assert (a - b).abs().max() < 0.002
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_constant_yield_earns_the_coupon():
    y = pd.Series(5.0, index=pd.date_range("2000-01-31", periods=24, freq="ME"))
    r = data.bond_total_return(y)
    assert np.allclose(r.to_numpy(), 0.05 / 12)


def test_rising_yield_loses_money():
    y = pd.Series([5.0, 6.0], index=pd.date_range("2000-01-31", periods=2, freq="ME"))
    assert data.bond_total_return(y).iloc[0] < -0.08


def test_to_real_deflates_and_drops_missing_inflation():
    df = pd.DataFrame({"stock": [0.02, 0.02], "bond": [0.0, 0.0], "bill": [0.01, 0.01],
                       "infl": [np.nan, 0.01]},
                      index=pd.date_range("1957-01-31", periods=2, freq="ME"))
    r = data.to_real(df)
    assert len(r) == 1
    assert r["bill"].iloc[0] == pytest.approx(0.0)
    assert r["stock"].iloc[0] == pytest.approx(1.02 / 1.01 - 1)


def test_provenance_lists_every_tape_with_a_pin():
    p = data.provenance()
    assert len(p) == 4 and all(len(r["sha256"]) == 16 for r in p)


@needs_monthly
def test_real_monthly_is_pinned_and_complete():
    m = data.load_monthly()
    assert m.index[0] == pd.Timestamp(data.START)
    assert m.index[-1] == pd.Timestamp(data.AS_OF)
    assert m.index.is_monotonic_increasing and not m.index.has_duplicates
    assert m[["stock", "bond", "bill"]].notna().all().all()
    assert m["infl"].first_valid_index() == pd.Timestamp(data.REAL_START)
    # long-run sanity: stocks > bonds > bills, compounded
    g = (1 + m[["stock", "bond", "bill"]]).prod() ** (12 / len(m)) - 1
    assert g["stock"] > g["bond"] > g["bill"] > 0
    assert data.fingerprint(m) == data.fingerprint(data.load_monthly())


@needs_monthly
def test_real_bond_leg_duration_vs_reprice():
    from quantlab import bundled
    aaa = bundled.load_arch("default")["AAA"]
    a = data.bond_total_return(aaa)
    b = data.bond_total_return(aaa, method="reprice")
    assert (a - b).abs().max() < 0.001


@needs_real
def test_real_daily_is_price_only_and_cut_at_asof():
    d = data.load_daily()
    assert d.index[-1] <= pd.Timestamp(data.AS_OF)
    assert d.index[0] >= pd.Timestamp("1990-01-02")
    assert d["bill"].between(-0.001, 0.001).all()
    assert d["stock"].abs().max() < 0.25
