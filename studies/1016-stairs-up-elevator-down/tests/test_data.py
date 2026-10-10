"""Data-layer tests for Study 1016 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stairs import data  # noqa: E402
from quantlab import bundled  # noqa: E402

needs_real = pytest.mark.skipif(
    not data.have_real(),
    reason="bundled tapes not available (skfolio cache absent) — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_returns(2000, 1.0, seed=1016)
    b, _ = data.synthetic_returns(2000, 1.0, seed=1016)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_returns(2000, 1.0, seed=1016)
    b, _ = data.synthetic_returns(2000, 1.0, seed=1017)
    assert not np.allclose(a["logret"].to_numpy(), b["logret"].to_numpy())


def test_synthetic_shape_index_and_horizon():
    df, truth = data.synthetic_returns(3000, 1.0, seed=1016)
    assert len(df) == 3000 == truth["n_days"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index[-1] < pd.Timestamp("2262-01-01")
    assert (df["price"] > 0).all()
    assert np.allclose(np.log(df["price"]).diff().dropna(), df["logret"].iloc[1:])
    assert np.allclose(df["ret"], np.expm1(df["logret"]))


def test_signal_strength_scales_leverage_and_holds_persistence():
    _, t1 = data.synthetic_returns(500, 1.0)
    _, th = data.synthetic_returns(500, 0.5)
    _, t0 = data.synthetic_returns(500, 0.0)
    assert t0["gamma_eff"] == 0.0
    assert t1["gamma_eff"] > th["gamma_eff"] > 0.0
    assert t1["persistence"] == pytest.approx(t0["persistence"])
    assert th["persistence"] == pytest.approx(t0["persistence"])
    assert t0["persistence"] < 1.0


def test_null_is_sign_symmetric():
    """With gamma = 0 the shock-for-shock mirror world has identical volatility."""
    p = data.PLANT
    T = 1500
    z = np.random.default_rng(3).standard_normal((T + 10, 1))
    a = data.simulate_gjr(T, 0.0, p["omega"], 0.08, 0.0, p["beta"], np.inf, shocks=z, burn=10)
    b = data.simulate_gjr(T, 0.0, p["omega"], 0.08, 0.0, p["beta"], np.inf, shocks=-z, burn=10)
    assert np.allclose(a, -b)


def test_leverage_breaks_the_mirror():
    p = data.PLANT
    T = 1500
    z = np.random.default_rng(3).standard_normal((T + 10, 1))
    a = data.simulate_gjr(T, 0.0, p["omega"], 0.01, 0.15, p["beta"], np.inf, shocks=z, burn=10)
    b = data.simulate_gjr(T, 0.0, p["omega"], 0.01, 0.15, p["beta"], np.inf, shocks=-z, burn=10)
    assert not np.allclose(np.abs(a), np.abs(b))


def test_simulators_are_vectorised_and_finite():
    y = data.simulate_gjr(400, 0.0, 1e-6, 0.02, 0.1, 0.9, 6.0, n_paths=5, seed=1)
    e = data.simulate_egarch(400, 0.0, -0.1, 0.1, -0.1, 0.98, 6.0, n_paths=5, seed=1)
    assert y.shape == (400, 5) and e.shape == (400, 5)
    assert np.isfinite(y).all() and np.isfinite(e).all()


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_returns(500, 1.0, seed=1)
    b, _ = data.synthetic_returns(500, 1.0, seed=2)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


# --------------------------------------------------------------------------- #
# Cash and calendar helpers
# --------------------------------------------------------------------------- #
def test_daily_rf_compounds_back_to_the_monthly_rate():
    ff = pd.DataFrame({"mkt": [0.01, -0.02], "rf": [0.003, 0.004]},
                      index=pd.to_datetime(["2000-01-31", "2000-02-29"]))
    idx = pd.bdate_range("2000-01-03", "2000-02-29")
    rf = data.daily_rf(idx, ff)
    jan = (1 + rf[rf.index.month == 1]).prod() - 1
    feb = (1 + rf[rf.index.month == 2]).prod() - 1
    assert jan == pytest.approx(0.003)
    assert feb == pytest.approx(0.004)


def test_daily_rf_is_nan_beyond_the_cash_tape():
    ff = pd.DataFrame({"mkt": [0.01], "rf": [0.003]}, index=pd.to_datetime(["2000-01-31"]))
    rf = data.daily_rf(pd.bdate_range("2000-01-03", "2000-02-10"), ff)
    assert rf[rf.index.month == 2].isna().all()


def test_ff_index_starts_at_one_and_compounds():
    ff = pd.DataFrame({"mkt": [0.10, -0.10], "rf": [0.0, 0.0]},
                      index=pd.to_datetime(["2000-01-31", "2000-02-29"]))
    lvl = data.ff_index(ff)
    assert lvl.iloc[0] == 1.0
    assert lvl.iloc[-1] == pytest.approx(1.1 * 0.9)


def test_as_of_dates_are_full_periods_and_not_in_the_future():
    for d in (data.SP500_AS_OF, data.NASDAQ_AS_OF, data.FF_AS_OF):
        ts = pd.Timestamp(d)
        assert ts == ts + pd.offsets.MonthEnd(0)
        assert ts < pd.Timestamp.today()
    assert data.AS_OF == max(data.SP500_AS_OF, data.NASDAQ_AS_OF, data.FF_AS_OF)


def test_provenance_pins_match_the_bundled_registry():
    assert data.PROVENANCE["sp500"]["sha256"] == bundled.SKFOLIO_FILES["sp500_index"][1]
    assert data.PROVENANCE["nasdaq"]["sha256"] == bundled.ARCH_FILES["nasdaq"][2]
    assert data.PROVENANCE["ff"]["sha256"] == bundled.ARCH_FILES["frenchdata"][2]
    assert "price" in data.PROVENANCE["sp500"]["kind"]
    assert "total return" in data.PROVENANCE["ff"]["kind"]


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_sp500_tape_is_cut_at_the_last_full_month():
    s = data.load_sp500()
    assert s.index[0] == pd.Timestamp("1990-01-02")
    assert s.index[-1] <= pd.Timestamp(data.SP500_AS_OF)
    assert s.index[-1] >= pd.Timestamp("2022-11-25")
    assert s.index.is_monotonic_increasing and not s.index.has_duplicates
    assert (s > 0).all() and len(s) > 8000


@needs_real
def test_nasdaq_and_ff_tapes_load():
    n = data.load_nasdaq()
    assert n.index[0] == pd.Timestamp("1999-01-04")
    assert n.index[-1] == pd.Timestamp("2018-12-31")
    ff = data.load_ff_market()
    assert ff.index[0] == pd.Timestamp("1926-07-31")
    assert ff.index[-1] == pd.Timestamp(data.FF_AS_OF)
    assert ff["mkt"].abs().max() < 0.5          # decimal, not percent
    assert (ff["rf"] >= -0.001).all()


@needs_real
def test_daily_rf_covers_the_trading_window():
    s = data.load_sp500()
    s = s[s.index <= pd.Timestamp(data.FF_AS_OF)]
    rf = data.daily_rf(s.index)
    assert rf.notna().all()
    assert 0.0 <= rf.mean() * 252 < 0.08
