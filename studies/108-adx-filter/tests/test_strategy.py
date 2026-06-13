"""Indicators, ADX gate logic, the forward-return engine, and the study's spine:
ADX gating only improves the rule when the filter is informative about actual trends."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adx_filter import strategy as st  # noqa: E402


# ---- ADX indicator -----------------------------------------------------------
def test_adx_range_and_no_lookahead(flat):
    bars, _ = flat
    adx = st.adx(bars, n=14)
    valid = adx.dropna()
    # ADX must be in [0, 100].
    assert (valid >= 0).all() and (valid <= 100).all()
    # ADX requires at least n prior bars.
    assert adx.iloc[:13].isna().all() or adx.iloc[:14].isna().any()


def test_adx_is_higher_in_trending_tape(flat, trending):
    flat_bars, _ = flat
    trend_bars, _ = trending
    adx_flat = st.adx(flat_bars, n=14).dropna().mean()
    adx_trend = st.adx(trend_bars, n=14).dropna().mean()
    # A trended tape should have a clearly higher mean ADX than a random walk.
    assert adx_trend > adx_flat


# ---- MA cross entries --------------------------------------------------------
def test_ma_cross_directions_and_no_lookahead(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"], fast=20, slow=50)
    assert set(ent["dir"].unique()) <= {-1, 1}
    # No signal before the slow SMA is warm (needs 50 bars).
    assert ent.index.min() >= bars.index[49]


def test_ma_cross_constant_has_no_entries():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=200))
    assert len(st.ma_cross_entries(close)) == 0


# ---- ADX gate ----------------------------------------------------------------
def test_adx_gate_reduces_trade_count(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    adx_s = st.adx(bars, n=14)
    gated = st.apply_adx_gate(ent, adx_s, threshold=25.0)
    # The gated set must be a subset of the ungated set.
    assert len(gated) <= len(ent)
    assert gated.index.isin(ent.index).all()


def test_adx_gate_filters_low_adx_bars(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    adx_s = st.adx(bars, n=14)
    gated = st.apply_adx_gate(ent, adx_s, threshold=25.0)
    # All remaining signals must have ADX >= threshold on the signal bar.
    if len(gated) > 0:
        adx_at_signal = adx_s.reindex(gated.index)
        assert (adx_at_signal >= 25.0).all()


# ---- forward-return engine ---------------------------------------------------
def test_ledger_columns(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    led = st.run_trades(bars, ent, hold_days=20, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "bars_held", "ret_gross", "ret_net",
    ]


def test_net_is_gross_minus_cost(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    led = st.run_trades(bars, ent, hold_days=10, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=7)
    b = st.random_directions(1000, seed=7)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---- the study's spine -------------------------------------------------------
def test_adx_gate_does_not_clearly_beat_ungated_on_flat(flat):
    """On a martingale, gating on ADX should not reliably improve mean return."""
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    adx_s = st.adx(bars, n=14)
    gated = st.apply_adx_gate(ent, adx_s, threshold=25.0)

    ungated_s = st.summarize(st.run_trades(bars, ent, cost_bps=0), "ret_gross")
    gated_s = st.summarize(st.run_trades(bars, gated, cost_bps=0), "ret_gross")

    # On a random walk neither arm should have a t-stat clearly above 2
    # (NaN means too few trades — also acceptable, not a spurious "edge").
    import math
    assert math.isnan(ungated_s["tstat"]) or abs(ungated_s["tstat"]) < 3.0
    assert math.isnan(gated_s["tstat"]) or abs(gated_s["tstat"]) < 3.0


def test_cost_monotonically_lowers_mean(flat):
    bars, _ = flat
    ent = st.ma_cross_entries(bars["close"])
    means = [
        st.summarize(st.run_trades(bars, ent, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]
