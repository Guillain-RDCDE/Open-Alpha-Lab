"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unstaked import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=960)
    b, _ = data.synthetic_panel(seed=960)
    for col in ("coin", "cheap", "dear", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_panel(seed=960)
    b, _ = data.synthetic_panel(seed=961)
    assert not np.allclose(a["coin"].to_numpy(), b["coin"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=2, seed=960)
    assert {"coin", "cheap", "dear", "cash", "coin_true"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 2 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay far inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert (prices[["coin", "cheap", "dear", "cash"]] > 0).all().all()


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=960)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_equalises_the_wrappers():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=960)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=960)
    assert abs(t0["drag_dear_planted"] - t0["drag_cheap_planted"]) < 1e-12
    assert t1["drag_dear_planted"] > t1["drag_cheap_planted"]
    assert t1["spread_planted_pct"] == pytest.approx(2.25, abs=1e-9)


def test_observed_coin_is_a_mistimed_version_of_the_truth():
    """The vendor coin close must differ from the true level day by day but track it."""
    prices, _ = data.synthetic_panel(seed=960)
    ratio = prices["coin"] / prices["coin_true"]
    assert ratio.std(ddof=1) > 0.0
    # the mis-timing is transient, not a drift: the log ratio is centred on zero
    assert abs(float(np.log(ratio).mean())) < 0.02


def test_synthetic_daily_alias_matches_panel():
    a, ta = data.synthetic_daily(seed=960)
    b, tb = data.synthetic_panel(seed=960)
    assert np.allclose(a["cheap"].to_numpy(), b["cheap"].to_numpy())
    assert ta["seed"] == tb["seed"]


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=960)
    b, _ = data.synthetic_panel(seed=961)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_unmeasured_cohort_is_named():
    """Coverage is a claim: the six spot-ETH lines this study does NOT measure are listed."""
    assert len(data.COHORT_NOT_MEASURED) >= 5
    assert not (set(data.COHORT_NOT_MEASURED) & set(data.FUNDS))


def test_declared_assumptions_are_present_and_sane():
    """The non-tape constants must exist, be labelled, and be swept over a real range."""
    assert set(data.EXPENSE_RATIOS) == {"ETHA", "ETHE"}
    assert 0.0 < data.EXPENSE_RATIOS["ETHA"] < data.EXPENSE_RATIOS["ETHE"] < 0.05
    assert data.ETHA_FEE_RANGE[0] < data.ETHA_FEE_RANGE[1]
    assert 0.0 < data.STAKING_YIELD_ANN < 0.10
    assert min(data.STAKING_SWEEP) < data.STAKING_YIELD_ANN < max(data.STAKING_SWEEP)


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (offline / CI) — "
                           "synthetic tests cover the logic")
def test_real_cache_measurement_runs():
    px = data.load_prices()
    assert px.index[0] >= pd.Timestamp(data.ETF_ERA_START)
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    td = st.tracking_difference(px["ETHE"], px["ETHA"])
    assert np.isfinite(td["ann_td_pct"]) and np.isfinite(td["monthly_t"])
    # fund-vs-fund noise must be far smaller than fund-vs-coin noise (the study's spine)
    td_coin = st.tracking_difference(px["ETHA"], px["ETH-USD"])
    assert td["daily_sd_bps"] < td_coin["daily_sd_bps"]
