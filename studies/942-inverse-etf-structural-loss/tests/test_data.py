"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_tax import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=942)
    b, _ = data.synthetic_daily(seed=942)
    for col in ("index_tr", "index_px", "fund", "cash", "rate_pct"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=10, seed=942)
    assert {"index_tr", "index_px", "fund", "cash", "rate_pct"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' nanosecond horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_seeds_differ():
    a, _ = data.synthetic_daily(seed=942)
    b, _ = data.synthetic_daily(seed=943)
    assert not np.allclose(a["fund"].to_numpy(), b["fund"].to_numpy())


def test_price_leg_lags_total_return_when_index_pays_a_dividend():
    prices, truth = data.synthetic_daily(div_yield_ann=0.02, seed=942)
    r_tr = prices["index_tr"].pct_change().dropna()
    r_px = prices["index_px"].pct_change().dropna()
    yield_ann = float((r_tr - r_px).mean() * data.TRADING_DAYS_PER_YEAR)
    assert abs(yield_ann - 0.02) < 2e-3
    assert truth["div_yield_ann"] == 0.02


def test_signal_strength_scales_the_planted_tax():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=942)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=942)
    _, th = data.synthetic_daily(signal_strength=0.5, seed=942)
    assert t0["tax_ann"] == 0.0
    assert t1["tax_ann"] == pytest.approx(t1["tax_ann_full"])
    assert th["tax_ann"] == pytest.approx(0.5 * t1["tax_ann_full"])
    assert t0["expected_gap_ann"] == pytest.approx(0.0)


def test_synthetic_panel_is_a_list_of_distinct_tapes():
    panel = data.synthetic_panel(seeds=(942, 943, 944))
    assert len(panel) == 3
    funds = [p["fund"].to_numpy() for p, _ in panel]
    assert not np.allclose(funds[0], funds[1])
    assert all(t["signal_strength"] == 1.0 for _, t in panel)


# --------------------------------------------------------------------------- #
# The ^IRX -> per-bar cash-rate conversion
# --------------------------------------------------------------------------- #
def test_cash_rate_daily_uses_previous_close_and_act360():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    irx = pd.Series([2.0, 4.0, 6.0], index=idx)
    rf = data.cash_rate_daily(irx)
    assert np.isnan(rf.iloc[0])
    # 3 Jan: one calendar day at the 2 Jan level of 2%.
    assert rf.iloc[1] == pytest.approx(0.02 * 1 / 360.0)
    # 6 Jan: three calendar days (weekend) at the 3 Jan level of 4%.
    assert rf.iloc[2] == pytest.approx(0.04 * 3 / 360.0)


def test_cash_rate_annualises_near_the_quoted_level():
    prices, truth = data.synthetic_daily(rate_ann=0.05, seed=942)
    rf = data.cash_rate_daily(prices["rate_pct"]).dropna()
    ann = float(rf.mean() * data.TRADING_DAYS_PER_YEAR)
    assert abs(ann - 0.05) < 2e-3  # act/365.25 over act/360 is a ~1.5% relative wedge


# --------------------------------------------------------------------------- #
# Cache-path convention & offline guarantees
# --------------------------------------------------------------------------- #
def test_cache_path_strips_caret_and_uses_shared_convention():
    p = data._cache_path("^IRX", os.sep + "tmp", "prices")
    assert os.path.basename(p) == "prices_IRX_1d.parquet"
    q = data._cache_path("SPY", os.sep + "tmp", "praw")
    assert os.path.basename(q) == "praw_SPY_1d.parquet"


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=942)
    b, _ = data.synthetic_daily(seed=943)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fund_table_is_consistent_with_expense_table():
    assert set(data.FUNDS) == set(data.EXPENSE_RATIO_PCT)
    for fund, (index, k) in data.FUNDS.items():
        assert k < 0 and index in ("SPY", "QQQ")


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="shared _cache absent (offline / CI) - synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    r = st.race(px, fund="SH", index="SPY", k=-1.0)
    for key in ("gap_ann_pct", "gap_t", "rf_ann_pct"):
        assert np.isfinite(r[key])
    # The two arms must track each other closely: a -1x fund and a -1x book are the
    # same exposure, so their return correlation is near-perfect.
    assert r["legs"]["r_fund"].corr(r["r_direct"]) > 0.99
