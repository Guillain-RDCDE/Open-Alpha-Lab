"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_trend import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=912)
    b, _ = data.synthetic_daily(seed=912)
    assert np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=20, seed=912)
    assert {"asset", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 20 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, truth = data.synthetic_daily(seed=912)
    assert (prices["cash"].diff().dropna() > 0).all()
    # ~cash_rate over the horizon
    grew = prices["cash"].iloc[-1] / prices["cash"].iloc[0]
    assert grew > 1.0


def test_signal_strength_zero_flattens_regimes():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=912)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=912)
    # At ss=0 the two regime vols collapse to (nearly) the same effective vol.
    assert abs(t0["vol_bull_eff"] - t0["vol_bear_eff"]) < 1e-9
    assert t1["vol_bear_eff"] > t1["vol_bull_eff"]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=912)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_daily(seed=913)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    gold, cash = px["GLD"].dropna(), px["BIL"].dropna()
    common = gold.index.intersection(cash.index)
    cmp = st.compare(gold.loc[common], cash.loc[common])
    for k in ("excess_sharpe_adv", "t_diff_vs_bh", "dd_bh_abs", "dd_trend_abs"):
        assert np.isfinite(cmp[k])
    # buy-and-hold gold is more volatile than the trend overlay (it is always invested)
    assert cmp["bh"]["vol_ann"] > cmp["trend"]["vol_ann"]
