"""The oscillators must match their textbook definitions and be causal (no look-ahead)."""

import numpy as np
import pandas as pd

from true_strength import oscillators as osc


def _series():
    rng = np.random.default_rng(0)
    return pd.Series(np.exp(np.cumsum(0.01 * rng.standard_normal(500))) + 10.0,
                     index=pd.bdate_range("2020-01-01", periods=500))


def test_tsi_in_bounds_and_zero_on_flat():
    s = _series()
    t = osc.tsi(s)["tsi"]
    assert (t.abs() <= 100.0 + 1e-6).all()           # TSI lives in [-100, 100]
    flat = pd.Series(np.full(300, 42.0), index=s.index[:300])
    assert osc.tsi(flat)["tsi"].abs().max() < 1e-6   # a dead-flat series has zero strength


def test_rsi_matches_wilder_bounds():
    s = _series()
    r = osc.rsi(s, 14)
    assert (r >= 0).all() and (r <= 100).all()
    up = pd.Series(np.linspace(10, 60, 100), index=pd.bdate_range("2020-01-01", periods=100))
    assert osc.rsi(up, 14).iloc[-1] > 95             # a pure up-ramp pins RSI near 100


def test_macd_hist_is_line_minus_signal():
    s = _series()
    m = osc.macd(s)
    pd.testing.assert_series_equal(m["hist"], (m["macd"] - m["signal"]).rename("hist"))


def test_panel_zscores_are_standardised():
    s = _series()
    p = osc.oscillator_panel(s)
    for col in ["z_tsi", "z_macd", "z_rsi"]:
        assert abs(p[col].mean()) < 1e-9
        assert abs(p[col].std(ddof=0) - 1.0) < 1e-6


def test_tsi_is_causal():
    """Changing a future bar must not move an earlier TSI value."""
    s = _series()
    base = osc.tsi(s)["tsi"]
    bumped = s.copy()
    bumped.iloc[400] *= 1.5
    after = osc.tsi(bumped)["tsi"]
    pd.testing.assert_series_equal(base.iloc[:399], after.iloc[:399])


def test_grid_size():
    assert len(osc.tsi_grid()) == 4 * 3 * 2
