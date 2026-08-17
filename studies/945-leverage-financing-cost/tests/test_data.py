"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lev_financing import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=945)
    b, _ = data.synthetic_panel(seed=945)
    for col in a.columns:
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=10, seed=945)
    assert {"index", "cash", "rate", "fund_2", "fund_3"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rate_cycle_is_present():
    prices, truth = data.synthetic_panel(seed=945)
    half = len(prices) // 2
    assert prices["rate"].iloc[:half].mean() == pytest.approx(truth["rate_low_pct"])
    assert prices["rate"].iloc[half:].mean() == pytest.approx(truth["rate_high_pct"])


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=945)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_removes_spread_and_fee():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=945)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=945)
    assert t1["planted_spread_pct"] > 0 and t1["expense_ratio_pct"] > 0
    assert t0["planted_spread_pct"] == 0.0 and t0["expense_ratio_pct"] == 0.0


def test_levered_fund_tracks_index_roughly_l_times():
    prices, _ = data.synthetic_panel(seed=945)
    r_idx = prices["index"].pct_change().dropna()
    for L in (2, 3):
        r_f = prices[f"fund_{L}"].pct_change().dropna()
        assert abs(np.corrcoef(r_idx, r_f)[0, 1]) > 0.95
        assert abs(r_f.std() / r_idx.std() - L) < 0.15


def test_synthetic_daily_wrapper_is_two_x_only():
    prices, truth = data.synthetic_daily(seed=945)
    assert "fund_2" in prices.columns and "fund_3" not in prices.columns
    assert truth["leverages"] == (2,)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=945)
    b, _ = data.synthetic_panel(seed=946)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_assumptions_are_declared():
    """Every non-tape input is a named constant, so the sweeps can reach it."""
    assert set(data.FUNDS) == {"SSO", "UPRO"}
    assert set(data.EXPENSE_RATIO) == set(data.FUNDS)
    assert 0.0 < data.SPY_EXPENSE_RATIO < 0.5
    assert all(v > 0 for v in data.MARGIN_SPREADS.values())


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (offline / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_estimate_runs():
    px = data.load_prices().dropna()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    rets = px[["SSO", "UPRO", "SPY"]].pct_change().dropna()
    rate = px["IRX"].reindex(rets.index)
    for fund, L in data.FUNDS.items():
        e = st.implied_financing(rets[fund], rets["SPY"], L,
                                 expense_ratio=data.EXPENSE_RATIO[fund],
                                 bench_fee=data.SPY_EXPENSE_RATIO, rate_pct=rate)
        assert np.isfinite(e["implied_financing_pct"])
        assert abs(e["beta"] - L) < 0.1        # the wrapper really is L-times levered
        assert e["drag_ann_pct"] > 0.0         # leverage is never free
