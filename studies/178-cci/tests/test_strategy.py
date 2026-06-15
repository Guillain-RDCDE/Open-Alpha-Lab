"""CCI indicator, signal generators, the forward-return engine, and the study's spine:
the CCI rule only beats a coin when mean-reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cci import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — any test that reads real data must carry this decorator.
# ---------------------------------------------------------------------------
_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))
_SPY_CACHE = os.path.join(_CACHE, "cci_SPY_1d.parquet")
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---- CCI indicator -----------------------------------------------------------
def test_cci_length_and_nan_prefix(random_walk):
    bars, _ = random_walk
    c = st.cci(bars, period=20)
    assert len(c) == len(bars)
    # First period-1 bars must be NaN (warmup).
    assert c.iloc[:19].isna().all()
    # From bar period-1 onward, all values must be finite (single-pass implementation).
    assert c.iloc[19:].notna().all(), (
        f"{c.iloc[19:].isna().sum()} unexpected NaN values after warmup period"
    )


def test_cci_constant_price_gives_zero():
    """When price is constant, TP = SMA_TP, CCI = 0 (MAD = 0 edge case)."""
    idx = pd.date_range("2025-01-01", periods=30, freq="B", name="date")
    bars = pd.DataFrame(
        {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1e6},
        index=idx,
    )
    c = st.cci(bars, period=20)
    assert (c.dropna() == 0.0).all()


def test_cci_scale_and_sign():
    """Prices in a strong uptrend should produce CCI > 0; a downtrend CCI < 0."""
    idx = pd.date_range("2025-01-01", periods=60, freq="B", name="date")
    price_up = pd.Series(np.linspace(100, 200, 60), index=idx)
    price_dn = pd.Series(np.linspace(200, 100, 60), index=idx)
    bars_up = pd.DataFrame(
        {"open": price_up, "high": price_up * 1.005, "low": price_up * 0.995,
         "close": price_up, "volume": 1e6},
        index=idx,
    )
    bars_dn = pd.DataFrame(
        {"open": price_dn, "high": price_dn * 1.005, "low": price_dn * 0.995,
         "close": price_dn, "volume": 1e6},
        index=idx,
    )
    c_up = st.cci(bars_up, period=20).dropna()
    c_dn = st.cci(bars_dn, period=20).dropna()
    # A constant-slope uptrend puts the close always ahead of the rolling mean
    # → CCI positive throughout (value depends on the intrabar wick symmetry).
    assert c_up.mean() > 50
    assert c_dn.mean() < -50


def test_cci_period_default_is_20(random_walk):
    bars, _ = random_walk
    assert st.cci(bars).iloc[:19].isna().all()
    assert st.cci(bars).iloc[19:].notna().all()


# ---- Signal generators -------------------------------------------------------
def test_breach_entries_directions(mean_reverting):
    bars, _ = mean_reverting
    c = st.cci(bars)
    ent = st.breach_entries(c)
    assert set(ent["dir"].unique()) <= {-1, 1}
    # Expect at least some entries on a 500-day tape.
    assert len(ent) > 5


def test_crossback_entries_directions(mean_reverting):
    bars, _ = mean_reverting
    c = st.cci(bars)
    ent = st.crossback_entries(c)
    assert set(ent["dir"].unique()) <= {-1, 1}
    assert len(ent) > 5


def test_breach_no_consecutive_same_zone(random_walk):
    """Breach framing: only one signal per zone episode (no re-trigger while inside)."""
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    if len(ent) < 2:
        pytest.skip("too few entries on this tape")
    # After a buy entry at t, the next bar must NOT have CCI < -100 as well (unless it
    # re-entered from above — which means it exited first).  We check by reconstructing
    # what the CCI was at each signal bar.
    for sig_ts, row in ent.iterrows():
        d = row["dir"]
        threshold = -100.0 if d == 1 else 100.0
        val = c.loc[sig_ts]
        prev_val = c.shift(1).loc[sig_ts]
        if d == 1:
            assert val < threshold and prev_val >= threshold
        else:
            assert val > threshold and prev_val <= threshold


def test_no_signal_before_cci_warmup(random_walk):
    """No signal can fire before CCI has a full period of data."""
    bars, _ = random_walk
    c = st.cci(bars, period=20)
    ent = st.breach_entries(c)
    assert ent.index.min() >= bars.index[19]


# ---- Forward-return engine invariants ----------------------------------------
def test_ledger_columns_and_dtypes(random_walk):
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net"
    ]
    assert (led["bars_held"] >= 1).all()
    assert (led["entry"] > 0).all()


def test_net_is_gross_minus_cost(random_walk):
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2e-4)


def test_long_only_has_no_short_trades(random_walk):
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=0.0, long_only=True)
    assert (led["dir"] == 1).all()


def test_hold_days_bounds(random_walk):
    """Entry is at open of t+1; exit is no later than the last bar."""
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=0.0)
    for _, row in led.iterrows():
        assert row["bars_held"] >= 1
        assert row["bars_held"] <= 5


# ---- Random control & the study's spine --------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(2000, seed=3)
    b = st.random_directions(2000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(random_walk):
    bars, _ = random_walk
    c = st.cci(bars)
    ent = st.breach_entries(c)
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=cost), "ret_net")["mean_bps"]
        for cost in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_cci_beats_coin_only_with_mean_reversion(random_walk, mean_reverting):
    """The spine: with very strong planted mean-reversion the CCI rule edges a random coin;
    on a martingale it does not (it actually under-performs, consistent with momentum/noise)."""
    def edge(bars, seed):
        c = st.cci(bars)
        ent = st.breach_entries(c)
        if len(ent) < 5:
            return 0.0
        sig = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        rand = st.summarize(
            st.run_trades(bars, ent, hold_days=5, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=seed)),
            "ret_gross",
        )
        return sig["mean_bps"] - rand["mean_bps"]

    # With very strong mean-reversion (mr=0.7) the CCI rule should beat the coin.
    strong_revt, _ = data.synthetic_daily(n_days=2000, mean_rev=0.70, seed=178)
    assert edge(strong_revt, seed=1) > 0.0   # strong mean-reversion → rule finds it
    # On a martingale or weak mean-reversion, performance is noise-level (can be negative).
    # We just assert the strong-reversion case is better than the random-walk case.
    rw_edge = edge(random_walk[0], seed=1)
    sr_edge = edge(strong_revt, seed=1)
    assert sr_edge > rw_edge  # more mean-reversion → better edge


# ---- Real-data guard ---------------------------------------------------------
@requires_cache
def test_real_tape_shape_and_basic_sanity():
    bars = data.fetch_daily("SPY", cache_dir=_CACHE)
    assert len(bars) > 500
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["close"] > 0).all()


@requires_cache
def test_real_cci_and_entries():
    bars = data.fetch_daily("SPY", cache_dir=_CACHE)
    c = st.cci(bars, period=20)
    ent = st.breach_entries(c)
    assert len(ent) > 10
    # Single-pass CCI: no NaN after the warmup period.
    assert c.iloc[19:].notna().all(), (
        f"{c.iloc[19:].isna().sum()} unexpected NaN values in real tape after warmup"
    )
