"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper_tax import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline, no cache)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=915)
    b, _ = data.synthetic_daily(seed=915)
    for col in ("k1", "ric", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=8, seed=915)
    assert {"k1", "ric", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 8 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_daily(seed=915)
    b, _ = data.synthetic_daily(seed=916)
    assert not np.allclose(a["ric"].to_numpy(), b["ric"].to_numpy())


def test_signal_strength_scales_the_planted_drag():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=915)
    _, t_half = data.synthetic_daily(signal_strength=0.5, seed=915)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=915)
    assert t1["planted_drag_ann"] == pytest.approx(t1["drag_ann"])
    assert t_half["planted_drag_ann"] == pytest.approx(0.5 * t1["drag_ann"])
    assert t0["planted_drag_ann"] == 0.0


def test_null_wrappers_share_the_factor_and_differ_only_by_noise(null):
    """At signal_strength=0 the two legs are the same economic asset plus noise."""
    prices, truth = null
    r = st.daily_returns(prices, ["ric", "k1"])
    d = r["ric"] - r["k1"]
    # The planted drift is exactly zero, so the realised drift is small relative to the
    # sampling error of the difference (the tracking noise dominates by construction).
    assert abs(d.mean() * 252) < 4.0 * (d.std(ddof=1) / np.sqrt(len(d)) * 252)
    assert r["ric"].corr(r["k1"]) > 0.9
    assert truth["planted_drag_ann"] == 0.0


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(seed=915)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_synthetic_panel_is_independent_and_deterministic():
    panel_a = data.synthetic_panel(n_pairs=3, signal_strength=0.0, seed=915)
    panel_b = data.synthetic_panel(n_pairs=3, signal_strength=0.0, seed=915)
    assert len(panel_a) == 3
    for (pa, _), (pb, _) in zip(panel_a, panel_b):
        assert np.allclose(pa["ric"].to_numpy(), pb["ric"].to_numpy())
    assert not np.allclose(panel_a[0][0]["ric"].to_numpy(), panel_a[1][0]["ric"].to_numpy())


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=915)
    b, _ = data.synthetic_daily(seed=916)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_expense_ratio_proxy_is_declared_for_both_wrappers():
    """The fee gap is a stated ASSUMPTION, not a tape measurement — but it must exist."""
    assert set(data.EXPENSE_RATIO_PROXY) == {data.WRAPPER_K1, data.WRAPPER_1099}
    assert data.EXPENSE_RATIO_PROXY[data.WRAPPER_K1] > data.EXPENSE_RATIO_PROXY[data.WRAPPER_1099]


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_without_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (fresh checkout / CI) — the "
                           "synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    res = st.race(px, data.WRAPPER_1099, data.WRAPPER_K1, data.CASH)
    for key in ("diff_ann", "t_diff", "te_ann", "sharpe_adv"):
        assert np.isfinite(res[key])
    # The two wrappers sit on the same commodity beta — they must be highly correlated.
    assert res["tracking"]["corr"] > 0.85
