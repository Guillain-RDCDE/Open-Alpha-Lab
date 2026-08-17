"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from short_pair import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=941)
    b, _ = data.synthetic_daily(seed=941)
    for col in ("under", "lev_long", "lev_short", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=10, seed=941)
    assert {"under", "lev_long", "lev_short", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_levered_legs_track_three_times_the_underlying():
    prices, _ = data.synthetic_daily(seed=941)
    r_u = prices["under"].pct_change().dropna()
    r_l = prices["lev_long"].pct_change().reindex(r_u.index)
    r_s = prices["lev_short"].pct_change().reindex(r_u.index)
    beta_l = np.polyfit(r_u, r_l, 1)[0]
    beta_s = np.polyfit(r_u, r_s, 1)[0]
    assert 2.9 < beta_l < 3.1
    assert -3.1 < beta_s < -2.9


def test_levered_legs_decay_versus_the_underlying():
    """Both legs must lose to L x the underlying's *compounded* return (the decay)."""
    prices, truth = data.synthetic_daily(signal_strength=0.0, seed=941)  # free funds
    total_u = prices["under"].iloc[-1] / prices["under"].iloc[0]
    total_l = prices["lev_long"].iloc[-1] / prices["lev_long"].iloc[0]
    # A 3x fund on a rising, volatile tape ends far below the naive cube of the index.
    assert total_l < total_u ** 3
    assert truth["decay_long_ann"] > 0.05


def test_signal_strength_scales_the_fee_load_not_the_decay():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=941)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=941)
    assert t1["expected_harvest_ann"] > 0.02
    assert t0["expected_harvest_ann"] == 0.0
    # The decay is identical in both worlds — only the fee load moves.
    assert t1["decay_long_ann"] == pytest.approx(t0["decay_long_ann"])
    assert t1["decay_short_ann"] == pytest.approx(t0["decay_short_ann"])


def test_synthetic_panel_is_distinct_across_seeds():
    panel = data.synthetic_panel(n_seeds=3, signal_strength=1.0)
    assert len(panel) == 3
    first = panel[0][0]["under"].to_numpy()
    assert not np.allclose(first, panel[1][0]["under"].to_numpy())
    assert all(t["expected_harvest_ann"] > 0.02 for _, t in panel)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=941)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_daily(seed=942)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_book_runs():
    px = data.load_prices().dropna()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    cmp = st.compare(px["TQQQ"], px["SQQQ"], px["BIL"], bench=px["QQQ"], rebalance="daily")
    for k in ("ann_mean", "sharpe", "t_hac", "max_drawdown"):
        assert np.isfinite(cmp[k])
    # The daily-reset pair short is (near) market-neutral by construction.
    assert abs(cmp["beta"]["beta"]) < 0.1
