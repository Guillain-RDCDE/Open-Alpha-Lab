"""Tests for the research-grade analytics (HAC, Lo SE, time-norm, capacity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import analytics, decompose, diagnostics


def _strong_overnight(seed=0, n=252 * 20):
    return decompose.decompose(
        diagnostics.synthetic_ohlc(
            overnight_bias_bps=6, intraday_bias_bps=0, daily_vol_bps=70, seed=seed, n_days=n
        )
    )


def test_hac_tstat_significant_for_strong_drift():
    dec = _strong_overnight()
    res = analytics.mean_tstat_hac(dec["r_overnight"])
    assert res["tstat"] > 3
    assert res["lags"] >= 1 and res["n"] == len(dec)


def test_hac_tstat_near_zero_for_no_drift():
    dec = decompose.decompose(
        diagnostics.synthetic_ohlc(overnight_bias_bps=0, intraday_bias_bps=0, seed=11, n_days=252 * 20)
    )
    res = analytics.mean_tstat_hac(dec["r_overnight"])
    assert abs(res["tstat"]) < 2.5  # not significant


def test_lo_sharpe_se_tstat_consistent():
    dec = _strong_overnight(seed=1)
    s = analytics.sharpe_with_se(dec["r_overnight"])
    # t-stat is annualisation-invariant: sharpe_ann/se_ann == tstat
    assert abs(s["tstat"] - s["sharpe_ann"] / s["se_ann"]) < 1e-9
    assert s["tstat"] > 2


def test_calendar_hours_weekday_and_weekend():
    # Thu, Fri, Mon, Tue -> overnight before Mon spans the weekend.
    idx = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
    h = analytics.calendar_hours(idx)
    assert np.isnan(h.iloc[0])
    assert abs(h.iloc[1] - 17.5) < 1e-9  # Thu->Fri, one night
    assert abs(h.iloc[2] - 65.5) < 1e-9  # Fri->Mon, weekend
    assert abs(h.iloc[3] - 17.5) < 1e-9


def test_time_normalized_summary_shape():
    dec = _strong_overnight(seed=2)
    t = analytics.time_normalized_summary(dec)
    assert list(t.index) == ["overnight", "intraday"]
    assert t.loc["intraday", "session_hours"] == 6.5
    assert t.loc["overnight", "session_hours"] > 6.5
    assert {"mean_bps_per_session", "session_hours", "mean_bps_per_hour"} <= set(t.columns)


def test_rolling_sharpe_length():
    dec = _strong_overnight(seed=3)
    window = 252 * 5
    rs = analytics.rolling_sharpe(dec["r_overnight"], window=window)
    assert len(rs) == len(dec["r_overnight"].dropna()) - window + 1
    assert rs.notna().all()


def test_capacity_positive_for_positive_edge():
    ohlc = diagnostics.synthetic_ohlc(overnight_bias_bps=5, intraday_bias_bps=0, seed=4)
    ohlc["Volume"] = 1_000_000.0
    dec = decompose.decompose(ohlc)
    cap = analytics.capacity_estimate(dec, ohlc)
    assert cap["capacity_usd"] > 0
    assert 0 < cap["participation_rate"] < 1


def test_capacity_curve_monotonic_and_crosses_zero():
    ohlc = diagnostics.synthetic_ohlc(overnight_bias_bps=5, intraday_bias_bps=0, seed=4)
    ohlc["Volume"] = 1_000_000.0
    dec = decompose.decompose(ohlc)
    cc = analytics.capacity_curve(dec, ohlc, sizes_usd=(1e5, 1e6, 1e7, 1e8, 1e9))
    # bigger size -> more impact -> lower net edge (strictly decreasing)
    net = cc["net_edge_bps"].values
    assert all(net[i] > net[i + 1] for i in range(len(net) - 1))
    # smallest size keeps most of the edge; largest destroys it
    assert net[0] > 0 > net[-1]
