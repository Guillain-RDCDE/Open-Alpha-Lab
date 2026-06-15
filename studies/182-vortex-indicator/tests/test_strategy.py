"""Strategy tests — the Vortex Indicator computation, crossover detection, the backtest
engine's invariants, the random control, and the study's spine.

All tests run offline (no network, no cache files).  Cache-gated tests skip on CI.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vortex_indicator import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache gate
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
_SAMPLE_CACHE = data._cache_path("SPY", _CACHE_DIR)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Vortex Indicator — mathematical properties
# ---------------------------------------------------------------------------
def test_vortex_columns_and_length(martingale):
    bars, _ = martingale
    vi = st.vortex(bars, period=14)
    assert list(vi.columns) == ["vi_plus", "vi_minus"]
    assert len(vi) == len(bars)


def test_vortex_nan_in_warmup(martingale):
    """The first ``period`` rows must be NaN (rolling sum over period bars not yet full)."""
    bars, _ = martingale
    vi = st.vortex(bars, period=14)
    # First period rows should be NaN (window not yet full after shifting)
    assert vi["vi_plus"].iloc[:14].isna().all()
    assert vi["vi_plus"].iloc[14:].notna().all()


def test_vortex_values_positive(martingale):
    """VI+ and VI- are ratios of strictly positive quantities — must be > 0 everywhere valid."""
    bars, _ = martingale
    vi = st.vortex(bars, period=14)
    valid_plus = vi["vi_plus"].dropna()
    valid_minus = vi["vi_minus"].dropna()
    assert (valid_plus > 0).all()
    assert (valid_minus > 0).all()


def test_vortex_symmetric_random_walk():
    """On a symmetric random walk the time-averaged VI+/VI- spread is near zero.

    We average across multiple seeds to average out finite-sample noise from the
    random-walk realisations.  The pooled mean difference should be small relative
    to the scale of VI (~1.0).
    """
    diffs = []
    for seed in range(10):
        bars, _ = data.synthetic_daily(n_days=500, momentum=0.0, seed=seed)
        vi = st.vortex(bars, period=14).dropna()
        diffs.append(float((vi["vi_plus"] - vi["vi_minus"]).mean()))
    mean_diff = float(np.mean(diffs))
    # Pooled across 10 seeds the expected mean should be close to 0
    assert abs(mean_diff) < 0.05


def test_vortex_period_sensitivity(martingale):
    """Shorter period should produce more crossovers due to faster oscillation."""
    bars, _ = martingale
    vi14 = st.vortex(bars, period=14)
    vi7 = st.vortex(bars, period=7)
    ent14 = st.crossover_entries(vi14)
    ent7 = st.crossover_entries(vi7)
    assert len(ent7) >= len(ent14)


# ---------------------------------------------------------------------------
# Crossover signals
# ---------------------------------------------------------------------------
def test_crossover_directions_are_only_plus_minus_one(martingale):
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    assert set(ent["dir"].unique()) <= {-1, 1}


def test_crossover_both_directions_present(martingale):
    """A random-walk tape of decent length must produce both up- and down-crossovers."""
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    assert (ent["dir"] == 1).sum() >= 1
    assert (ent["dir"] == -1).sum() >= 1


def test_crossover_min_gap_reduces_entries(martingale):
    """Setting a positive min_gap must produce fewer or equal entries than gap=0."""
    bars, _ = martingale
    vi = st.vortex(bars)
    all_ent = st.crossover_entries(vi, min_gap=0.0)
    filtered = st.crossover_entries(vi, min_gap=0.05)
    assert len(filtered) <= len(all_ent)


def test_crossover_no_lookahead(martingale):
    """No crossover signal can appear before the VI warm-up period is complete."""
    bars, _ = martingale
    vi = st.vortex(bars, period=14)
    ent = st.crossover_entries(vi)
    # First valid VI value is at index 14 (0-indexed), signal requires prior bar too
    first_valid_idx = vi["vi_plus"].first_valid_index()
    assert ent.index.min() >= first_valid_idx


def test_constant_price_has_no_crossovers():
    """A completely flat price produces no range → VI is undefined → no signals."""
    idx = pd.bdate_range("2020-01-02", periods=100, name="date")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6},
        index=idx,
    )
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    assert len(ent) == 0


# ---------------------------------------------------------------------------
# Backtest engine invariants
# ---------------------------------------------------------------------------
def test_ledger_columns(martingale):
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    led = st.run_trades(bars, ent, hold_days=14, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net",
    ]


def test_net_is_gross_minus_cost(martingale):
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    led = st.run_trades(bars, ent, hold_days=10, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_cost_monotonically_lowers_mean(martingale):
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=14, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_hold_days_respected(martingale):
    """Each trade must hold for at most ``hold_days`` bars."""
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    hold = 5
    led = st.run_trades(bars, ent, hold_days=hold, cost_bps=0.0)
    assert (led["bars_held"] <= hold).all()
    assert (led["bars_held"] >= 1).all()


# ---------------------------------------------------------------------------
# Random control
# ---------------------------------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=5)
    b = st.random_directions(1000, seed=5)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---------------------------------------------------------------------------
# The study's spine — the VI cross beats a coin only with planted momentum
# ---------------------------------------------------------------------------
def test_vi_beats_coin_only_with_momentum():
    """Core thesis: VI crossover edges a random-direction coin only when momentum exists.

    On a martingale (momentum=0) the cross should not systematically beat the coin;
    on a trending tape (momentum=0.30) the cross should show a positive average edge.

    Because each individual tape is a single random realisation with few crossovers,
    we pool 12 seeds for each regime and check that the *average* edge is:
      - positive and meaningfully so on the trending tape, and
      - centred near zero on the martingale.
    This mirrors the desk's standard: the machinery should detect what it is designed
    to detect when it is planted, and read zero when it is not.
    """
    def edge_one(momentum, seed):
        bars, _ = data.synthetic_daily(n_days=2000, momentum=momentum, seed=seed)
        vi = st.vortex(bars, period=14)
        ent = st.crossover_entries(vi)
        if len(ent) < 5:
            return None
        cross = st.summarize(
            st.run_trades(bars, ent, hold_days=14, cost_bps=0), "ret_gross"
        )
        rand = st.summarize(
            st.run_trades(bars, ent, hold_days=14, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=99 + seed)),
            "ret_gross",
        )
        return cross["mean_bps"] - rand["mean_bps"]

    trend_edges = [e for e in (edge_one(0.30, s) for s in range(12)) if e is not None]
    flat_edges  = [e for e in (edge_one(0.00, s) for s in range(12)) if e is not None]

    # Trending tape: average cross advantage over coin must be positive
    assert np.mean(trend_edges) > 0.0
    # Martingale: average is centred near zero — not persistently one-sided
    assert abs(np.mean(flat_edges)) < abs(np.mean(trend_edges))


def test_summarize_empty_ledger():
    """summarize() must handle an empty ledger gracefully (return nan, not crash)."""
    led = pd.DataFrame(columns=["entry_ts", "dir", "entry", "exit_price",
                                 "bars_held", "ret_gross", "ret_net"])
    s = st.summarize(led, "ret_net")
    assert all(np.isnan(v) for v in s.values())


def test_summarize_contains_required_keys(martingale):
    bars, _ = martingale
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    led = st.run_trades(bars, ent, hold_days=14, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    for k in ("n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"):
        assert k in s


# ---------------------------------------------------------------------------
# Real-data integration — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_data_produces_entries_and_ledger():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    vi = st.vortex(bars, period=14)
    ent = st.crossover_entries(vi)
    assert len(ent) > 0
    led = st.run_trades(bars, ent, hold_days=14, cost_bps=1.0)
    assert len(led) > 0
    assert led["ret_net"].notna().all()


@requires_cache
def test_real_data_tstat_is_finite():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_CACHE_DIR)
    vi = st.vortex(bars)
    ent = st.crossover_entries(vi)
    led = st.run_trades(bars, ent, hold_days=14, cost_bps=0)
    s = st.summarize(led, "ret_gross")
    assert np.isfinite(s["tstat"])
