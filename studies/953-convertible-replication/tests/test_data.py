"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_repl import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=953)
    b, _ = data.synthetic_panel(seed=953)
    for col in a.columns:
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=12, seed=953)
    assert {"eq", "growth", "credit", "cash", "fund"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=953)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_kills_both_planted_effects():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=953)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=953)
    assert t1["alpha_ann"] > 0 and t1["convexity"] > 0
    assert t0["alpha_ann"] == 0.0 and t0["convexity"] == 0.0


def test_seed_changes_the_tape():
    a, _ = data.synthetic_panel(seed=953)
    b, _ = data.synthetic_panel(seed=954)
    assert not np.allclose(a["fund"].to_numpy(), b["fund"].to_numpy())


def test_factor_correlations_are_planted():
    prices, _ = data.synthetic_panel(seed=953)
    r = prices[["eq", "growth", "credit"]].pct_change().dropna()
    c = r.corr()
    assert c.loc["eq", "growth"] > c.loc["eq", "credit"]
    assert c.loc["eq", "growth"] > 0.8


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=953)
    fp1 = data.fingerprint(a)
    b, _ = data.synthetic_panel(seed=954)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_synthetic_daily_alias_matches_panel():
    a, _ = data.synthetic_daily(seed=953)
    b, _ = data.synthetic_panel(seed=953)
    assert np.allclose(a["fund"].to_numpy(), b["fund"].to_numpy())


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_fee_assumptions_are_labelled_constants():
    """The expense ratios are a documented PROXY, not tape — keep them explicit."""
    assert set(data.FUND_FEE_ANN) == set(data.FUNDS)
    assert 0.0 < data.REPLICA_FEE_ANN < 0.01


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_holdout_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    sub = px[["CWB", "SPY", "QQQ", "LQD", "BIL"]].dropna()
    res = st.holdout(sub, "CWB", ("SPY", "QQQ", "LQD"), "BIL", data.IS_END)
    assert np.isfinite(res["oos"]["alpha_ann"])
    assert np.isfinite(res["oos"]["t_hac"])
    # the fitted mix must be a real long-only portfolio inside its budget
    w = res["fit"]["w_vec"]
    assert (w >= -1e-9).all() and w.sum() <= 1.0 + 1e-9
