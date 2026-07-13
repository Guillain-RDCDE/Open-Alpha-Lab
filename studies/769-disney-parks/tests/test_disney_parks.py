"""Fully offline, deterministic tests for Study 769 — no network, fixed seeds.

Assert pure-function behaviour on synthetic/hardcoded data:
  * the risk/return primitives on a known path,
  * the release lag is real (year-Y attendance is not public until ~July Y+1),
  * the synthetic positive control: t stays small at edge=0 and lights up at edge>0,
  * the annual-excess t is finite and signed on a constructed pair.

Run: pytest -q studies/769-disney-parks/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from disney_parks import data, strategy as st


def test_cagr_and_mdd_on_known_path():
    # A level that doubles over exactly 2 years -> CAGR ~ sqrt(2)-1 ~ 41.42%.
    idx = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"])
    lvl = pd.Series([100.0, 141.421356, 200.0], index=idx)
    assert abs(st.cagr(lvl) - (2 ** 0.5 - 1)) < 1e-3
    # Monotone up -> zero drawdown.
    assert st.max_drawdown(lvl) == 0.0
    # A dip to 50 from 100 -> -50% drawdown.
    dip = pd.Series([100.0, 50.0, 80.0], index=idx)
    assert abs(st.max_drawdown(dip) - (-0.5)) < 1e-9


def test_release_lag_is_real():
    # Year-Y attendance must NOT be public before ~July Y+1 (strict no-look-ahead).
    rel = data.release_date(2019)
    assert rel.year == 2020 and rel.month == data.THEME_INDEX_RELEASE_MONTH
    growth = data.attendance_growth()
    g2018 = float(growth[growth.index.year == 2018].iloc[0])
    g2019 = float(growth[growth.index.year == 2019].iloc[0])
    idx = pd.to_datetime(["2012-06-30", "2012-08-31", "2020-06-30", "2020-08-31"])
    stepped = data._stepped_signal(idx, growth.rename("pg"))
    # Before ANY release (first growth is 2011, public July-2012) -> NaN, no look-ahead.
    assert np.isnan(stepped.loc["2012-06-30"])
    assert np.isfinite(stepped.loc["2012-08-31"])
    # In mid-2020, 2019's figure is NOT yet public (released July-2020): must still show 2018's.
    assert abs(stepped.loc["2020-06-30"] - g2018) < 1e-9
    assert abs(stepped.loc["2020-08-31"] - g2019) < 1e-9


def test_synthetic_control_null_and_planted():
    # edge=0 must NOT manufacture significance; a planted edge MUST light up.
    t_null = st.control_recovers(data.synthetic(edge=0.0), 0.0)["t"]
    t_edge = st.control_recovers(data.synthetic(edge=0.02), 0.02)["t"]
    assert abs(t_null) < 2.0
    assert t_edge > 2.5
    # Deterministic: same seed -> identical stat.
    assert st.control_recovers(data.synthetic(edge=0.0), 0.0)["t"] == t_null


def test_annual_excess_t_finite_and_signed():
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2015, 2022)])
    # a compounds faster than b -> positive mean excess, positive t.
    a = pd.Series(100.0 * 1.20 ** np.arange(7), index=idx)
    b = pd.Series(100.0 * 1.05 ** np.arange(7), index=idx)
    out = st.annual_excess_t(a, b)
    assert out["n"] == 6 and np.isfinite(out["t"]) and out["mean_excess"] > 0 and out["t"] > 0


def test_lead_lag_zero_signal_is_null():
    # A constant (zero-variance) signal cannot predict anything -> NaN/finite, never a spurious hit.
    idx = pd.period_range("2010-01", periods=120, freq="M").to_timestamp(how="end")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"dis": 100 * np.exp(np.cumsum(rng.normal(0.005, 0.05, 120))),
         "spy": 100 * np.exp(np.cumsum(rng.normal(0.005, 0.04, 120))),
         "pg": np.full(120, 3.0)},
        index=idx,
    )
    out = st.lead_lag(frame, "pg", horizon=12, lag=1, excess=False)
    # Constant regressor -> slope not identified; the HAC t must not be a finite |t|>=2 hit.
    assert not (np.isfinite(out["t"]) and abs(out["t"]) >= 2.0)
