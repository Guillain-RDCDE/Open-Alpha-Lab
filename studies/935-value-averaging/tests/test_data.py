"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value_avg import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=6, seed=935)
    b, _ = data.synthetic_daily(n_years=6, seed=935)
    assert np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=8, seed=935)
    assert {"asset", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 8 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(n_years=6, seed=935)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_kills_the_wobble():
    _, t1 = data.synthetic_daily(n_years=6, signal_strength=1.0, seed=935)
    _, t0 = data.synthetic_daily(n_years=6, signal_strength=0.0, seed=935)
    assert t0["swing_eff"] == 0.0
    assert t1["swing_eff"] > 0.0


def test_signal_strength_creates_mean_reversion():
    """The planted wobble must push the long-horizon variance ratio below one.

    A variance ratio ``Var(12-month) / (12 * Var(1-month))`` under 1 is the standard
    signature of transitory, reverting price moves — the thing a contrarian schedule
    can harvest. A random walk sits at ~1.
    """
    def vr(prices, q=12):
        r = np.log(prices["asset"]).diff().dropna().to_numpy()
        m = r[: (len(r) // 21) * 21].reshape(-1, 21).sum(axis=1)
        long = m[: (len(m) // q) * q].reshape(-1, q).sum(axis=1)
        return float(long.var(ddof=1) / (q * m.var(ddof=1)))
    a1, _ = data.synthetic_daily(n_years=25, signal_strength=1.0, seed=935)
    a0, _ = data.synthetic_daily(n_years=25, signal_strength=0.0, seed=935)
    assert vr(a1) < vr(a0)
    assert vr(a1) < 0.9


def test_synthetic_panel_paths_are_distinct():
    panel = data.synthetic_panel(n_paths=3, signal_strength=0.0, n_years=4)
    assert len(panel) == 3
    fps = {data.fingerprint(p) for p, _ in panel}
    assert len(fps) == 3


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(n_years=5, seed=935)
    b, _ = data.synthetic_daily(n_years=5, seed=936)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_month_ends_are_last_trading_day_of_each_month():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    me = data.month_ends(idx)
    assert len(me) == 6
    assert me[0] == pd.Timestamp("2020-01-31")
    # every month-end must be the maximum date within its own month
    for d in me:
        same = idx[(idx.year == d.year) & (idx.month == d.month)]
        assert d == same.max()


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    a, c = px["SPY"].dropna(), px["BIL"].dropna()
    k = a.index.intersection(c.index)
    df = st.rolling_race(a.loc[k], c.loc[k], horizon_months=36)
    assert len(df) > 100
    s = st.summarise(df, horizon_months=36)
    for key in ("gap_mean_cents", "t_hac", "va_win_rate", "irr_prog_va", "irr_eq_va"):
        assert np.isfinite(s[key])
    # Both arms commit the same capital, so any wealth difference is the rule, not the funding.
    assert np.allclose(df["committed"].to_numpy(), df["committed"].iloc[0])
