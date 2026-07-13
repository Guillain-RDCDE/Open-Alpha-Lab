"""Fully offline, deterministic tests for Study 742 (Friday-17th).

No network: every test runs on the synthetic tape or on pure calendar arithmetic.
Asserts pure-function results, so the machinery is guarded independently of the live
Italian tapes.

    pytest -q studies/742-friday-17th/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from friday_17th import data as dt, strategy as st  # noqa: E402


def test_friday_17th_calendar_arithmetic():
    # 2027-09-17 is a Friday the 17th; 2027-09-10 (Fri) and 2027-08-17 (Tue) are not.
    idx = pd.DatetimeIndex(["2027-09-17", "2027-09-10", "2027-08-17", "2021-12-17"])
    mask = dt.is_friday_17th(idx)
    assert mask.tolist() == [True, False, False, True]  # 2021-12-17 is also a Friday


def test_synthetic_null_does_not_fire():
    # effect=0 -> Venerdi 17 is just another Friday; the detector must stay near zero.
    ts = np.array([st.synthetic_detect(0.0, seed=742 + s)["t"] for s in range(20)])
    assert (np.abs(ts) >= 2).sum() <= 2          # at most the ordinary false-positive rate
    assert abs(ts.mean()) < 0.6


def test_synthetic_planted_fear_is_recovered():
    # A planted NEGATIVE Venerdi-17 bump must show up as a strongly negative t.
    s = st.synthetic_detect(-1.0, seed=742)
    assert s["t"] < -4.0
    assert s["mean"] < 0.0
    assert s["n"] > 30                            # enough synthetic events to be meaningful


def test_one_sample_t_matches_hand_computation():
    x = np.array([0.01, -0.02, 0.03, -0.01, 0.00])
    r = st.one_sample_t(x)
    exp_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert r["n"] == 5
    assert abs(r["t"] - exp_t) < 1e-9


def test_short_is_the_negated_return():
    # On the synthetic tape, shorting the 17th must earn the negated day return.
    close, _ = dt.synthetic_daily(f17_effect=-1.0, seed=742)
    df = st.build_frame(close)
    f17_ret = df.loc[df["is_f17"], "ret"].to_numpy()
    sh = st.short_the_17th(close, cost_bps=0.0, borrow_bps=0.0)
    assert abs(sh["gross_mean_bps"] - (-f17_ret.mean() * 1e4)) < 1e-6
    # With a planted fear the short is profitable gross; costs only subtract.
    assert sh["gross_mean_bps"] > 0


def test_down_day_rate_wilson_bounds():
    x = np.array([-0.01, -0.02, 0.03, -0.01, 0.02, 0.00 - 1e-9])
    r = st.down_day_rate(x)
    assert 0.0 <= r["lo"] <= r["rate"] <= r["hi"] <= 1.0
    assert r["k"] == int((x < 0).sum())
