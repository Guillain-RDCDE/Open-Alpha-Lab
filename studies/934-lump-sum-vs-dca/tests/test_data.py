"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lump_vs_dca import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=934)
    b, _ = data.synthetic_daily(seed=934)
    assert np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())


def test_synthetic_seed_changes_the_path():
    a, _ = data.synthetic_daily(seed=934)
    b, _ = data.synthetic_daily(seed=935)
    assert not np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=20, seed=934)
    assert {"asset", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 20 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(seed=934)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_scales_the_planted_excess_drift():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=934)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=934)
    _, tm = data.synthetic_daily(signal_strength=-1.0, seed=934)
    assert t1["excess_drift_eff"] > 0 > tm["excess_drift_eff"]
    assert t0["excess_drift_eff"] == 0.0
    # Realised drift tracks the planted one in the right order.
    assert t1["realised_excess_ann"] > t0["realised_excess_ann"] > tm["realised_excess_ann"]


def test_synthetic_panel_shape_and_bond_is_calmer():
    prices, truth = data.synthetic_panel(n_years=15, seed=934)
    assert {"equity", "bond", "cash"}.issubset(prices.columns)
    eq_vol = prices["equity"].pct_change().std()
    bd_vol = prices["bond"].pct_change().std()
    assert bd_vol < eq_vol
    assert truth["n_days"] == 15 * data.TRADING_DAYS_PER_YEAR


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=934)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_daily(seed=935)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    spy, bil = px["SPY"].dropna(), px["BIL"].dropna()
    common = spy.index.intersection(bil.index)
    exp = st.run_experiment(spy.loc[common], bil.loc[common], cost_bps=1.0)
    assert len(exp) > 100
    s = st.summarise(exp, n_boot=200)
    for k in ("win_rate", "mean_gap_cents", "t_hac", "dispersion_ratio"):
        assert np.isfinite(s[k])
    # DCA lands in a tighter band than the lump sum — that much of the folklore is true.
    assert s["dispersion_ratio"] < 1.0
