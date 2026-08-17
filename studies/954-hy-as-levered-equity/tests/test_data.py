"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hy_replication import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=954)
    b, _ = data.synthetic_panel(seed=954)
    for col in ("fund", "equity", "duration", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=12, seed=954)
    assert {"fund", "equity", "duration", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_daily_alias_matches_panel():
    a, ta = data.synthetic_daily(n_years=6, seed=954)
    b, tb = data.synthetic_panel(n_years=6, seed=954)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta["w_true"] == tb["w_true"]


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=954)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_only_moves_the_shock_mean():
    """The two worlds differ in the shock's *price*, never in its size or the blend."""
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=954)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=954)
    assert t1["resid_vol"] == t0["resid_vol"]
    assert t1["w_true"] == t0["w_true"]
    assert t1["shock_mean_ann"] < 0.0 < t0["shock_mean_ann"]
    assert t0["shock_mean_ann"] == pytest.approx(t0["fair_premium_ann"])


def test_fair_premium_levels_the_sharpe_by_construction():
    """At signal_strength=0 the fund's ex-ante Sharpe equals the blend's, by design."""
    _, t = data.synthetic_panel(signal_strength=0.0, seed=954)
    mu_blend = (t["w_true"] * t["mu_equity"] + (1 - t["w_true"]) * t["mu_duration"]
                - t["cash_rate_ann"])
    sharpe_blend = mu_blend / t["vol_blend_ann"]
    sharpe_fund = (mu_blend + t["shock_mean_ann"]) / t["vol_fund_ann"]
    assert sharpe_fund == pytest.approx(sharpe_blend, rel=1e-9)


def test_fund_is_more_volatile_than_its_blend():
    _, t = data.synthetic_panel(seed=954)
    assert t["vol_fund_ann"] > t["vol_blend_ann"]


def test_expense_ratio_table_is_a_labelled_proxy():
    """The quoted fees are a documented PROXY, not tape — keep them explicit and sane."""
    for tk in ("HYG", "JNK", "USHY", "SPY", "IEF"):
        assert 0.0 < data.EXPENSE_RATIO_PCT[tk] < 1.0
    assert data.EXPENSE_RATIO_PCT["HYG"] > data.EXPENSE_RATIO_PCT["SPY"]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=954)
    fp1 = data.fingerprint(a)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    b, _ = data.synthetic_panel(seed=955)
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_ticker_lists_are_consistent():
    assert set(data.TICKERS) == (
        set(data.HY_TICKERS) | set(data.LEG_TICKERS) | set(data.DUR_LEG_TICKERS)
    )
    assert "BIL" in data.LEG_TICKERS  # the cash leg every arm is raced excess-of
    # The duration-leg sweep must stay Treasury-only: a credit-bearing leg (AGG, BND,
    # LQD) would put the very risk under test inside the benchmark.
    assert not ({"AGG", "BND", "LQD"} & set(data.DUR_LEG_TICKERS))
    assert "IEF" not in data.DUR_LEG_TICKERS  # the headline leg, swept against these


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    d = px[["HYG", "SPY", "IEF", "BIL"]].dropna()
    cmp = st.compare(d["HYG"], d["SPY"], d["IEF"], d["BIL"])
    for k in ("excess_sharpe_gap", "t_gap", "r2", "residual_ann"):
        assert np.isfinite(cmp[k])
    # HY sits between the two legs in volatility, as a credit blend must.
    assert 0.0 < cmp["w_mean"] < 1.0
