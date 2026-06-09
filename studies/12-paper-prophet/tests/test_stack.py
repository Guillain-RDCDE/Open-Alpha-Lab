"""The faithful port must behave like the article's stack — and never peek at the future."""

import numpy as np
import pandas as pd

from paper_prophet.stack import LOOKBACK, SIZE_CAP, VOL_FLOOR, generate_signals


def test_walkforward_shape_and_columns(wf):
    assert {"ret", "forecast", "sign", "vol", "size"} <= set(wf.frame.columns)
    assert len(wf.frame) == 40
    assert wf.lookback == LOOKBACK


def test_size_is_capped_and_floored(wf):
    # position size is min(1, 1/sigma) with sigma floored at 0.1 -> size in (0, 1].
    s = wf.frame["size"].dropna()
    assert (s <= SIZE_CAP + 1e-12).all()
    assert (s > 0).all()
    assert (s <= 1.0 / VOL_FLOOR + 1e-9).all()


def test_no_lookahead_grade_date_after_window(returns):
    # The day being graded must sit strictly after the 252-day window that produced its forecast.
    wf = generate_signals(returns, lookback=252, max_steps=5)
    first_grade = wf.frame.index[0]
    # the window for the first graded day is returns[0:252]; the graded day is returns[252].
    assert first_grade == returns.index[252]


def test_constant_long_uses_plus_one_but_same_sizes(returns):
    base = generate_signals(returns, lookback=252, max_steps=20)
    # constant_long must not change the recorded forecasts/sizes — only how returns combine them.
    cl = generate_signals(returns, lookback=252, max_steps=20, constant_long=True)
    pd.testing.assert_frame_equal(base.frame, cl.frame)
    # the constant-long stack return is +size*ret everywhere (no sign flips).
    expected = (base.frame["size"] * base.frame["ret"])
    pd.testing.assert_series_equal(cl.stack_returns, expected, check_names=False)


def test_sign_is_in_minus_one_zero_plus_one(wf):
    assert set(np.unique(wf.frame["sign"])) <= {-1.0, 0.0, 1.0}
