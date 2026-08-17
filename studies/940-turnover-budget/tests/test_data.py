"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turnover_budget import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_panel_is_deterministic():
    a, ca, _ = data.synthetic_panel(n_assets=6, n_years=5, seed=940)
    b, cb, _ = data.synthetic_panel(n_assets=6, n_years=5, seed=940)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ca.to_numpy(), cb.to_numpy())


def test_synthetic_panel_seed_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=6, n_years=5, seed=940)
    b, _, _ = data.synthetic_panel(n_assets=6, n_years=5, seed=941)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_panel_shape_and_index():
    prices, cash, truth = data.synthetic_panel(n_assets=7, n_years=6, seed=940)
    assert prices.shape == (6 * data.TRADING_DAYS_PER_YEAR, 7)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert cash.index.equals(prices.index)
    assert truth["n_days"] == len(prices)
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert (prices > 0).all().all()


def test_signal_strength_scales_the_planted_alpha():
    _, _, t1 = data.synthetic_panel(signal_strength=1.0, seed=940)
    _, _, t0 = data.synthetic_panel(signal_strength=0.0, seed=940)
    _, _, th = data.synthetic_panel(signal_strength=0.5, seed=940)
    assert t0["alpha_vol_eff"] == 0.0
    assert t1["alpha_vol_eff"] > th["alpha_vol_eff"] > 0.0


def test_synthetic_cash_is_monotone_growing():
    _, cash, truth = data.synthetic_panel(n_years=5, seed=940)
    assert (cash.diff().dropna() > 0).all()
    assert cash.iloc[-1] / cash.iloc[0] > 1.0
    assert truth["cash_rate_ann"] > 0


def test_synthetic_daily_wrapper_shape():
    frame, truth = data.synthetic_daily(n_years=4, seed=940)
    assert {"asset", "cash"} == set(frame.columns)
    assert len(frame) == truth["n_days"]


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=940)
    b, _, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=941)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_ticker_universe_is_the_eleven_sectors_plus_cash():
    assert len(data.SECTORS) == 11
    assert data.CASH in data.TICKERS and data.BENCH in data.TICKERS


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_frequency_table_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    sec = px[list(data.SECTORS)].dropna(how="all")
    tbl = st.frequency_table(sec, px["BIL"].dropna(), freqs=("M", "Q"))
    assert np.isfinite(tbl["ann_turnover"]).all()
    # A slower clock must trade less. This is arithmetic, not a finding.
    assert tbl.loc["M", "ann_turnover"] > tbl.loc["Q", "ann_turnover"]
