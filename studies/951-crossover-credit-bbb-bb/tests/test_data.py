"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossover_credit import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=951)
    b, _ = data.synthetic_panel(seed=951)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_seeds_differ():
    a, _ = data.synthetic_panel(seed=951)
    b, _ = data.synthetic_panel(seed=952)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=10, seed=951)
    assert {"ig", "crossover", "hy", "dur", "eq", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert (prices > 0).all().all()


def test_synthetic_cash_is_monotone_growing():
    prices, truth = data.synthetic_panel(seed=951)
    assert (prices["cash"].diff().dropna() > 0).all()
    grew = prices["cash"].iloc[-1] / prices["cash"].iloc[0]
    assert grew > 1.0
    # ~cash_rate_ann compounded over the horizon
    assert grew == pytest.approx((1.0 + truth["cash_rate_ann"]) ** truth["n_years"], rel=0.02)


def test_signal_strength_scales_the_planted_alpha():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=951)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=951)
    _, th = data.synthetic_panel(signal_strength=0.5, seed=951)
    assert t0["alpha_planted"] == 0.0
    assert t1["alpha_planted"] == pytest.approx(t1["alpha_crossover"])
    assert th["alpha_planted"] == pytest.approx(0.5 * t1["alpha_planted"])


def test_signal_strength_only_moves_the_crossover_rung():
    """The null and the planted world must differ on the crossover rung alone."""
    full, _ = data.synthetic_panel(signal_strength=1.0, seed=951)
    null, _ = data.synthetic_panel(signal_strength=0.0, seed=951)
    assert np.allclose(full["ig"].to_numpy(), null["ig"].to_numpy())
    assert np.allclose(full["hy"].to_numpy(), null["hy"].to_numpy())
    assert not np.allclose(full["crossover"].to_numpy(), null["crossover"].to_numpy())


def test_synthetic_daily_is_an_alias_of_the_panel():
    a, _ = data.synthetic_daily(seed=951)
    b, _ = data.synthetic_panel(seed=951)
    assert list(a.columns) == list(b.columns)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_excess_returns_are_excess_of_the_cash_leg():
    prices, _ = data.synthetic_panel(seed=951)
    e = st.excess_returns(prices, cash="cash")
    r = st.daily_returns(prices).dropna()
    assert np.allclose(e["hy"].to_numpy(), (r["hy"] - r["cash"]).to_numpy())
    assert np.allclose(e["cash"].to_numpy(), 0.0)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=951)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_panel(seed=952)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_ladder_constants_are_coherent():
    assert data.RUNGS == ("AGG", "LQD", "ANGL", "HYG")
    assert set(data.CROSSOVER) | set(data.BROAD_HY) | set(data.FACTORS) | {data.CASH} <= set(data.TICKERS)
    assert set(data.RUNGS) <= set(data.TICKERS)


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_ladder_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    core = px[list(data.RUNGS) + list(data.FACTORS) + [data.CASH]].dropna()
    tbl = st.ladder_table(core, rungs=data.RUNGS, factors=data.FACTORS, cash=data.CASH)
    assert list(tbl.index) == list(data.RUNGS)
    assert np.isfinite(tbl["alpha_ann"].to_numpy()).all()
    # Duration loading falls, and equity beta rises, as you walk down the quality ladder.
    assert tbl.loc["LQD", "beta_dur"] > tbl.loc["HYG", "beta_dur"]
    assert tbl.loc["HYG", "beta_eq"] > tbl.loc["LQD", "beta_eq"]
