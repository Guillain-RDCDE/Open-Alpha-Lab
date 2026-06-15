"""Signal, portfolio construction, and the study's spine: rotation beats 50/50 only when
ratio momentum is planted, not on a martingale."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eth_btc_ratio import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate for real-data tests
# ---------------------------------------------------------------------------
_ETH_CACHE = data._cache_path(data.ETH_TICKER, data.DEFAULT_CACHE)
_BTC_CACHE = data._cache_path(data.BTC_TICKER, data.DEFAULT_CACHE)
_BOTH_CACHED = os.path.exists(_ETH_CACHE) and os.path.exists(_BTC_CACHE)

requires_cache = pytest.mark.skipif(
    not _BOTH_CACHED,
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------
def test_signal_values_in_valid_set(null_tape):
    bars, _ = null_tape
    sig = st.momentum_signal(bars, lookback=20)
    assert set(sig.dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_signal_no_lookahead(null_tape):
    """Signal on day t must not use prices after day t (built on ratio_log which uses close)."""
    bars, _ = null_tape
    sig = st.momentum_signal(bars, lookback=20)
    # First valid signal requires at least 'lookback' prior closes — so no signal in first 20 rows
    assert sig.iloc[:20].isna().all(), "Signal should be NaN in the first lookback days"
    assert sig.iloc[20:].notna().any(), "Signal should be non-NaN after lookback days"


def test_ratio_log_correct(null_tape):
    bars, _ = null_tape
    rl = st.ratio_log(bars)
    manual = np.log(bars["ETH-USD"]["close"] / bars["BTC-USD"]["close"])
    assert np.allclose(rl.to_numpy(), manual.to_numpy())


# ---------------------------------------------------------------------------
# Backtest engine tests
# ---------------------------------------------------------------------------
def test_backtest_columns(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    expected = {
        "open_ret_btc", "open_ret_eth", "signal", "weight_eth", "weight_btc",
        "port_gross", "cost", "port_net", "bnh_btc", "bnh_eth", "fifty_fifty",
    }
    assert set(bt.columns) == expected


def test_weights_sum_to_one(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    assert np.allclose(bt["weight_eth"] + bt["weight_btc"], 1.0)


def test_net_is_gross_minus_cost(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    diff = (bt["port_gross"] - bt["port_net"] - bt["cost"]).abs()
    assert diff.max() < 1e-12


def test_cost_monotonically_lowers_mean(null_tape):
    bars, _ = null_tape
    means = [
        st.backtest_rotation(bars, lookback=20, cost_bps=c)["port_net"].mean()
        for c in [0.0, 5.0, 10.0]
    ]
    assert means[0] > means[1] > means[2]


def test_zero_cost_gross_equals_net(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20, cost_bps=0.0)
    # With no cost the first row might still have zero cost from initial position
    assert np.allclose(bt["port_gross"], bt["port_net"])


def test_fifty_fifty_is_equal_weight(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    manual = 0.5 * bt["open_ret_btc"] + 0.5 * bt["open_ret_eth"]
    assert np.allclose(bt["fifty_fifty"].to_numpy(), manual.to_numpy())


# ---------------------------------------------------------------------------
# The study's spine: rotation beats 50/50 ONLY with planted momentum
# ---------------------------------------------------------------------------
def test_rotation_beats_baseline_only_with_momentum(null_tape, trending_tape):
    """On a martingale ratio, rotation does not systematically beat 50/50.
    On a tape with planted ratio momentum, it does."""
    def edge(bars):
        bt = st.backtest_rotation(bars, lookback=20, cost_bps=0.0)
        rot = bt["port_net"].mean()
        base = bt["fifty_fifty"].mean()
        return rot - base

    trending_edge = edge(trending_tape[0])
    null_edge = edge(null_tape[0])
    assert trending_edge > 0.0, f"Trending tape: expected positive edge, got {trending_edge:.6f}"
    # On a martingale the edge should be small (within coin-flip noise over 500 days)
    assert abs(null_edge) < 0.005, f"Null tape: edge should be near zero, got {null_edge:.6f}"


# ---------------------------------------------------------------------------
# Summary and alpha tests
# ---------------------------------------------------------------------------
def test_summarize_shape(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    s = st.summarize(bt["port_net"])
    for key in ("n", "mean_ann_pct", "sharpe_ann", "tstat", "max_dd_pct"):
        assert key in s
    assert s["n"] > 0
    assert np.isfinite(s["tstat"])


def test_alpha_vs_baseline_keys(null_tape):
    bars, _ = null_tape
    bt = st.backtest_rotation(bars, lookback=20)
    result = st.alpha_vs_baseline(bt["port_net"], bt["fifty_fifty"])
    for key in ("alpha_daily_bps", "alpha_ann_pct", "beta", "r_squared", "t_alpha"):
        assert key in result


def test_lookback_sweep_returns_dataframe(null_tape):
    bars, _ = null_tape
    df = st.lookback_sweep(bars, lookbacks=[10, 20, 30])
    assert len(df) == 3
    assert "lookback" in df.columns
    assert "sharpe_rotation" in df.columns
    assert "tstat_rotation" in df.columns


# ---------------------------------------------------------------------------
# Real-data tests — guarded by cache presence
# ---------------------------------------------------------------------------
@requires_cache
def test_real_backtest_runs_and_has_positive_n():
    bars = data.load_pair(fetch=False)
    bt = st.backtest_rotation(bars, lookback=20)
    assert len(bt) > 500


@requires_cache
def test_real_summarize_finite():
    bars = data.load_pair(fetch=False)
    bt = st.backtest_rotation(bars, lookback=20)
    s = st.summarize(bt["port_net"])
    assert np.isfinite(s["sharpe_ann"])
    assert np.isfinite(s["tstat"])
