"""The four ingredients: exact definitions, exact calendar, no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from ambush import signals


def _bars(rows, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_ibs_bounds_and_degenerate_bar():
    df = _bars([[10, 12, 8, 8.4], [10, 12, 8, 12], [10, 10, 10, 10]])
    out = signals.ibs(df)
    assert out.between(0, 1).all()
    assert out.iloc[0] == pytest.approx(0.1)  # closed at the bottom decile of the range
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(0.5)  # zero-range bar reads neutral


def test_low_ibs_threshold_is_inclusive():
    df = _bars([[10, 18, 8, 10], [10, 18, 8, 10.5]])  # IBS exactly 0.20, then 0.25
    flags = signals.low_ibs(df)
    assert bool(flags.iloc[0]) and not bool(flags.iloc[1])


def test_tom_mask_marks_last_1_and_first_3_trading_days():
    # Jan 2024: last trading day Wed 31; Feb starts Thu 1, Fri 2, Mon 5.
    idx = pd.bdate_range("2024-01-02", "2024-02-29")
    m = signals.tom_mask(idx)
    marked = set(m[m].index.strftime("%Y-%m-%d"))
    assert {"2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"} <= marked
    assert "2024-01-30" not in marked and "2024-02-06" not in marked
    assert "2024-02-28" not in marked and "2024-02-29" in marked  # leap-month end


def test_tom_tomorrow_is_the_calendar_shifted_not_lagged():
    idx = pd.bdate_range("2024-01-02", "2024-02-29")
    s = signals.tom_tomorrow(idx)
    # at the close of Jan 30 we know Jan 31 (day -1) is TOM -> armed on the 30th
    assert bool(s.loc["2024-01-30"])
    # at the close of Feb 5 (last TOM day) tomorrow is day +4 -> dark
    assert not bool(s.loc["2024-02-05"])
    assert not bool(s.iloc[-1])  # tomorrow is outside the sample


def test_red_day_and_vix_stress():
    df = _bars([[10, 11, 9, 10], [10, 11, 9, 9.5], [9.5, 11, 9, 10.5]])
    assert signals.red_day(df).tolist() == [False, True, False]

    flat = pd.Series(20.0, index=pd.bdate_range("2024-01-02", periods=40))
    spiked = flat.copy()
    spiked.iloc[-1] = 24.0  # 24 >= 1.15 * ~20.1
    assert not signals.vix_stress(flat).iloc[-1]
    assert signals.vix_stress(spiked).iloc[-1]
    assert not signals.vix_stress(flat).iloc[:19].any()  # no signal before the window exists


def test_confluence_counts_and_alignment():
    n = 60
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame(
        {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.0}, index=idx
    )
    # a single all-signals day: closes at its low, below yesterday
    df.loc[idx[40], "Close"] = 9.0
    vix = pd.Series(20.0, index=idx)
    vix.iloc[40] = 25.0
    out = signals.confluence(df, vix)
    assert set(out.columns) == set(signals.SIGNAL_COLS + ["count"])
    assert out["count"].iloc[40] >= 3
    assert (out["count"] == out[signals.SIGNAL_COLS].sum(axis=1)).all()


def test_vix_holiday_gap_is_ffilled_never_backfilled():
    idx = pd.bdate_range("2024-01-02", periods=30)
    spy = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.0}, index=idx)
    vix = pd.Series(20.0, index=idx.delete(10))  # VIX missing one SPY session
    out = signals.confluence(spy, vix)
    assert not out["s_vix"].isna().any()
