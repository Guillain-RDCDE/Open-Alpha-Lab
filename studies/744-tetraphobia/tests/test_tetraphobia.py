"""Fully offline, deterministic tests for Study 744 — Tetraphobia.

No network: every assertion is a pure-function result on synthetic or hand-built inputs.
Guards the two detectors' machinery and the inference primitives.

    pytest -q studies/744-tetraphobia/tests
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tetraphobia import data as dt, strategy as st  # noqa: E402


def test_trailing_digit_counts_pure():
    # prices ending .04, .08, .04, .18 -> digits 4, 8, 4, 8
    import pandas as pd
    s = pd.Series([10.04, 10.08, 20.04, 20.18], index=pd.date_range("2020-01-01", periods=4))
    counts = st.trailing_digit_counts({"X": s})
    assert counts[4] == 2 and counts[8] == 2
    assert counts.sum() == 4


def test_tetraphobia_stats_symmetric_is_null():
    # equal 4s and 8s -> z(8>4) exactly 0, share 0.5
    counts = np.zeros(10, dtype=np.int64)
    counts[4] = counts[8] = 500
    counts[1] = counts[2] = counts[3] = counts[6] = counts[7] = counts[9] = 500
    s = st.tetraphobia_stats(counts)
    assert abs(s["z8_gt_4"]) < 1e-9
    assert abs(s["share8"] - 0.5) < 1e-9


def test_tetraphobia_stats_detects_8_preference():
    counts = np.zeros(10, dtype=np.int64)
    counts[4], counts[8] = 400, 600      # 8 over-represented
    for d in (1, 2, 3, 6, 7, 9):
        counts[d] = 500
    s = st.tetraphobia_stats(counts)
    assert s["z8_gt_4"] > 2          # clears the bar
    assert s["z4"] < 0               # digit 4 below the non-round mean


def test_one_sample_t_and_wilson():
    x = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    lo, hi = st.wilson_interval(3, 5)
    assert 0.0 <= lo < hi <= 1.0


def test_synthetic_digit_null_is_quiet_planted_fires():
    # null (bias=0) should mostly sit near |z| < 2; a planted bias must light up
    nulls = np.array([st.synthetic_digit_detect(0.0, seed=744 + s)["z8_gt_4"] for s in range(20)])
    assert (np.abs(nulls) >= 2).sum() <= 3          # ~standard-normal false-positive rate
    assert st.synthetic_digit_detect(0.05)["z8_gt_4"] > 2


def test_synthetic_calendar_null_quiet_planted_dip_fires():
    nulls = np.array([st.synthetic_calendar_detect(0.0, seed=744 + s)["t"] for s in range(20)])
    assert (np.abs(nulls) >= 2).sum() <= 4
    assert st.synthetic_calendar_detect(-0.02)["t"] < -2   # planted crash detected


def test_date_session_returns_snaps_forward_and_no_nan():
    import pandas as pd
    # a tape with a gap over 4 April 2021 (a Sunday) -> snaps to the next session
    idx = pd.bdate_range("2021-01-01", "2021-12-31")
    close = pd.Series(100.0 + np.arange(len(idx)) * 0.1, index=idx)
    ev = st.date_session_returns(close, 4, 4, years=(2021,))
    assert len(ev) == 1
    d, r = ev[0]
    assert d >= pd.Timestamp("2021-04-04")   # on/after the calendar date
    assert np.isfinite(r)


def test_data_constants_wellformed():
    assert dt.CALENDAR_CORE == ["EWT", "EWH", "MCHI"]
    assert len(dt.all_cluster_tickers()) == 20
    assert set(dt.ASIA_CLUSTER.values()) == {"Taiwan", "HongKong", "ChinaA", "Korea"}
