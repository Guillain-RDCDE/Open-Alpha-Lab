"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td_persist import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, ta = data.synthetic_panel(seed=913)
    b, tb = data.synthetic_panel(seed=913)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta["fees_bps"] == tb["fees_bps"]


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_funds=5, n_years=12, seed=913)
    assert list(prices.columns) == [f"fund_{j}" for j in range(5)] + ["cash"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=913)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_flattens_the_fee_ladder():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=913)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=913)
    assert t1["fee_spread_bps_eff"] > 20.0
    assert abs(t0["fee_spread_bps_eff"]) < 1e-9
    assert len(set(round(v, 9) for v in t0["fees_bps"].values())) == 1


def test_cheapest_fund_really_is_cheapest():
    _, truth = data.synthetic_panel(signal_strength=1.0, seed=913)
    fees = truth["fees_bps"]
    assert min(fees, key=fees.get) == truth["cheapest"] == "fund_0"


def test_fee_ladder_shows_in_realised_returns():
    """The planted fee must actually appear as a drag in the generated tape."""
    prices, truth = data.synthetic_panel(signal_strength=1.0, n_years=25, seed=913)
    funds = [c for c in prices.columns if c != "cash"]
    cagr = {f: prices[f].iloc[-1] / prices[f].iloc[0] for f in funds}
    assert cagr["fund_0"] > cagr[funds[-1]]  # cheapest ends ahead of dearest


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=913)
    b, _ = data.synthetic_panel(seed=914)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_expense_ratios_are_labelled_assumptions_and_cover_the_panel():
    for tk in data.SP500_FUNDS + data.NDX_ETFS + data.UNAVAILABLE:
        assert tk in data.EXPENSE_RATIO_BPS
        assert 0.0 < data.EXPENSE_RATIO_BPS[tk] < 100.0
    # The families are leader-first (the flagship is the dearest of its family).
    assert data.SP500_FUNDS[0] == "SPY" and data.NDX_ETFS[0] == "QQQ"
    assert data.EXPENSE_RATIO_BPS["QQQ"] > data.EXPENSE_RATIO_BPS["QQQM"]


def test_price_only_note_is_explicit():
    note = data.price_only_note()
    assert "PRICE-ONLY" in note and "dividend" in note


def test_splg_is_declared_unavailable_not_silently_swapped():
    assert "SPLG" in data.UNAVAILABLE
    assert "SPLG" not in data.TICKERS


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_pipeline_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    p = px[list(data.SP500_ETFS)].dropna()
    ann = st.annual_returns(p)
    assert int(ann.index.max()) < pd.Timestamp(data.AS_OF).year  # no partial year
    td = st.tracking_difference(ann)
    assert np.allclose(td.sum(axis=1).to_numpy(), 0.0, atol=1e-6)
    g = st.gap_stats(p, data.EXPENSE_RATIO_BPS, "cheapest", "leader")
    assert np.isfinite(g["annual_gap_bp_yr"]) and np.isfinite(g["t_annual"])
