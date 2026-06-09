"""The crossover signal recovers the trend's golden>death spread (and finds nothing on the null), the
book uses only the past, beats buy-and-hold when there's a trend, and adds nothing on a random walk."""

import numpy as np

from fools_gold import data, cross, strategy


def test_deterministic_and_trend(trend):
    close, truth = trend
    close2, _ = data.synthetic_prices(trend_strength=0.0006, seed=21)
    assert np.allclose(close.to_numpy(), close2.to_numpy())
    assert truth.has_trend


def test_cross_state_is_signed(trend_close):
    st = cross.cross_state(trend_close).dropna()
    assert set(np.unique(st)).issubset({-1.0, 0.0, 1.0})


def test_signal_value_recovers_trend(trend_close):
    sv = cross.signal_value(trend_close)
    assert sv["spread_ann_pct"] > 3.0          # golden days out-return death days when there's a trend
    assert sv["signal_present"]


def test_signal_flat_on_null(null_close):
    sv = cross.signal_value(null_close)
    assert abs(sv["spread_ann_pct"]) < 5.0     # random walk: crossover state predicts ~nothing


def test_weights_past_only_long_flat(trend_close):
    w = strategy.timing_weights(trend_close)
    assert set(np.unique(w.dropna())).issubset({0.0, 1.0})   # long/flat
    st = (cross.cross_state(trend_close) > 0).astype(float)
    assert w.dropna().equals(st.shift(1).reindex(w.index).dropna())


def test_beats_buy_hold_on_trend(trend_close):
    cmp = strategy.compare(trend_close, cost_bps=2.0)
    assert cmp["sharpe_gain"] > 0.1
    assert cmp["turnover_ann"] < 10.0          # 50/200 flips only a few times a year


def test_no_edge_on_null(null_close):
    cmp = strategy.compare(null_close, cost_bps=2.0)
    assert cmp["sharpe_gain"] < 0.2


def test_cost_sweep_monotone(trend_close):
    sw = strategy.cost_sweep(trend_close)
    g = sw["sharpe_gain"].to_numpy()
    assert (np.diff(g) <= 1e-9).all()
