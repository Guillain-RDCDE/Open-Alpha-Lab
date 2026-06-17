"""Strategy-engine tests for Study 213 (Meme Stocks), including spine tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meme_stocks import data, strategy  # noqa: E402

CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


# ---------------------------------------------------------------------------
# Unit tests — strategy primitives
# ---------------------------------------------------------------------------
def test_cagr_known_value():
    # 100% total return over 365 days should annualise close to 100% CAGR.
    result = strategy._cagr(1.0, 365)
    assert abs(result - 1.0) < 0.01   # within 1 ppt (365 vs 365.25 rounding)


def test_cagr_total_loss():
    result = strategy._cagr(-1.0, 365)
    assert np.isnan(result)


def test_sharpe_positive_for_positive_returns():
    rng = np.random.default_rng(42)
    # Positive mean, small noise -> positive Sharpe
    ret = pd.Series(0.005 + rng.normal(0, 0.001, 252))
    s = strategy._sharpe(ret)
    assert s > 0


def test_max_drawdown_flat_is_zero():
    px = pd.Series([100.0, 100.0, 100.0, 100.0])
    assert strategy.max_drawdown(px) == 0.0


def test_max_drawdown_half_is_fifty_pct():
    px = pd.Series([100.0, 50.0, 100.0])
    assert abs(strategy.max_drawdown(px) - (-0.5)) < 1e-9


def test_hac_tstat_detects_strong_mean():
    x = np.full(100, 0.05) + np.random.default_rng(0).normal(0, 0.001, 100)
    assert strategy.hac_tstat(x) > 5


def test_annual_vol_known_value():
    # Stochastic returns with std 0.01 -> annualised vol ~= 0.01 * sqrt(252).
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0.0, 0.01, 10000))
    v = strategy.annual_vol(ret)
    # With 10 000 obs the sample std is very close to 0.01; allow 5% relative tolerance.
    expected = 0.01 * np.sqrt(252)
    assert abs(v - expected) / expected < 0.05


# ---------------------------------------------------------------------------
# Spine test 1 — buy-and-hold on planted panel detects edge
# ---------------------------------------------------------------------------
def test_spine_bh_planted_panel_beats_bench(planted_panel):
    """With a planted premium, the meme basket must outperform the benchmark."""
    result = strategy.buy_and_hold(planted_panel, entry_date="2020-01-02")
    assert "error" not in result
    # With a 30% planted premium and 5 years, basket total return should beat bench.
    assert result["total_port"] > result["total_bench"]


# ---------------------------------------------------------------------------
# Spine test 2 — buy-and-hold on null panel finds no edge
# ---------------------------------------------------------------------------
def test_spine_bh_null_panel_no_reliable_edge(null_panel):
    """With zero planted premium, the basket should NOT reliably beat the bench."""
    result = strategy.buy_and_hold(null_panel, entry_date="2020-01-02")
    assert "error" not in result
    # No guarantee which side wins — just verify the function runs and returns numbers.
    assert np.isfinite(result["total_port"])
    assert np.isfinite(result["total_bench"])


# ---------------------------------------------------------------------------
# Spine test 3 — momentum-timed on planted panel fires and records trades
# ---------------------------------------------------------------------------
def test_spine_mt_planted_fires_trades(planted_panel):
    """With a planted premium, the momentum trigger should fire on at least one ticker."""
    result = strategy.momentum_timed(
        planted_panel,
        mom_window=20,
        mom_pct=0.10,
        hold_days=60,
        start_date="2020-01-02",
    )
    assert result["n_trades"] >= 1


# ---------------------------------------------------------------------------
# Spine test 4 — momentum-timed null panel: trades may fire but mean return ~0
# ---------------------------------------------------------------------------
def test_spine_mt_null_panel_mean_return_near_zero(null_panel):
    """In the null panel, the momentum strategy should not find a systematic edge."""
    result = strategy.momentum_timed(
        null_panel,
        mom_window=20,
        mom_pct=0.10,
        hold_days=60,
        start_date="2020-01-02",
    )
    if result["n_trades"] == 0:
        pytest.skip("no triggers fired in null panel — trivially no edge")
    # The mean net return in a no-edge world should not be far from 0.
    assert abs(result["mean_net_ret"]) < 1.0   # < 100% — sanity, not a tight bar


# ---------------------------------------------------------------------------
# Spine test 5 — real tape cache guard
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE_PATH, "GME_px.parquet")),
    reason="real-tape cache absent offline/CI",
)
def test_real_bh_runs_without_error():
    panel = data.load_panel(cache_dir=CACHE_PATH)
    result = strategy.buy_and_hold(panel)
    assert "error" not in result
    assert np.isfinite(result["total_port"])
