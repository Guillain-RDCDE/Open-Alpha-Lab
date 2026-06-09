"""The two ways of writing the mean-reversion strategy — a constant RSI band and a σ landmark —
produce the *same trades*, because the σ transform only relabels. And positions are tradable
(shifted one bar, never look-ahead)."""

import numpy as np

from sigma_sleight import rsi as R
from sigma_sleight import signals


def test_sigma_band_equals_implied_constant_band(close):
    """The headline identity at the signal layer: σ-thresholding == constant-RSI-thresholding."""
    for n, lower_sigma in ((2, -1.732), (14, -1.0), (50, -2.14)):
        rsi = R.wilder_rsi(close, n)
        pos_sigma = signals.sigma_band_positions(rsi, n, lower_sigma, 0.0)
        lo, ex = signals.implied_rsi_band(n, lower_sigma, 0.0)
        pos_const = signals.rsi_band_positions(rsi, lo, ex)
        assert np.array_equal(pos_sigma.to_numpy(), pos_const.to_numpy())


def test_implied_band_is_below_50(close):
    """An oversold σ landmark implies a lower band under 50 and an exit at exactly 50."""
    lo, ex = signals.implied_rsi_band(2, -1.732, 0.0)
    assert lo < 50.0
    assert np.isclose(ex, 50.0)


def test_positions_are_long_or_flat_and_shifted(close):
    rsi = R.wilder_rsi(close, 2)
    pos = signals.rsi_band_positions(rsi, 10.0, 50.0)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}
    assert pos.iloc[0] == 0.0           # shifted: nothing on the first bar


def test_strategy_actually_trades(close):
    """On the mean-reverting tape an oversold band enters at least a few times."""
    rsi = R.wilder_rsi(close, 2)
    pos = signals.rsi_band_positions(rsi, 10.0, 50.0)
    assert (pos.diff() > 0).sum() > 5


def test_forward_return_shifts_back(close):
    fwd = signals.forward_return(close, 5)
    assert fwd.iloc[-5:].isna().all()
    assert np.isclose(fwd.iloc[0], close.iloc[5] / close.iloc[0] - 1.0)
