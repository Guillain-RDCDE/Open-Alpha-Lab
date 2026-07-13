"""Fully offline, deterministic tests for Study 737 (Sunspot-Cycle).

No network, no cache: every assertion runs on the hardcoded solar calendar, the labelled
proxy, or the seeded synthetic world. Run with ``pytest -q studies/737-sunspot-cycle/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sunspot_cycle import data, strategy as st  # noqa: E402


def test_solar_calendar_shape():
    cyc = data.solar_cycles()
    assert list(cyc["cycle"]) == list(range(16, 26))          # cycles 16..25
    tps = data.turning_points()
    assert len(tps) == 20                                     # 10 minima + 10 maxima
    assert set(tps["kind"]) == {"min", "max"}
    # every maximum falls after its own cycle's minimum
    for _, r in cyc.iterrows():
        assert r["max_date"] > r["min_date"]


def test_proxy_pins_turning_points():
    """Activity is ~0 at each minimum and ~1 at each maximum (the proxy's defining pins)."""
    idx = pd.date_range("1928-01-31", "2025-12-31", freq="ME")
    prox = data.sunspot_proxy(idx)
    assert prox["activity"].dropna().between(-1e-9, 1 + 1e-9).all()
    # near cycle 21 max (1979-12) activity must be close to 1; near its min (1976-03) close to 0
    near_max = prox.loc[(prox.index >= "1979-10-01") & (prox.index <= "1980-02-28"), "activity"]
    near_min = prox.loc[(prox.index >= "1976-01-01") & (prox.index <= "1976-06-30"), "activity"]
    assert near_max.max() > 0.97
    assert near_min.min() < 0.05


def test_one_sample_t_and_wilson():
    mean, t = st.one_sample_t(np.array([0.01, 0.02, 0.015, 0.012, 0.018]))
    assert mean > 0 and t > 2                                 # a clearly-positive constant-ish set
    lo, hi = st.wilson_interval(7, 9)
    assert 0.0 <= lo < 7 / 9 < hi <= 1.0


def test_forward_return_rejects_pre_tape_event():
    m = pd.Series(np.linspace(100, 200, 60),
                  index=pd.date_range("1990-01-31", periods=60, freq="ME"))
    # an event before the tape must NOT snap forward to the start (the 1923-min bug guard)
    assert st.forward_return(m, pd.Timestamp("1985-01-01"), 12) is None
    r = st.forward_return(m, pd.Timestamp("1990-06-01"), 12)
    assert r is not None and r > 0                            # a real in-tape forward return


def test_synthetic_null_does_not_fire_but_planted_does():
    """The regime/phase detector: quiet on amp=0, loud on a planted cycle (power control)."""
    fires = 0
    for seed in range(12):
        close, prox = data.synthetic_world(amp=0.0, seed=737 + seed)
        if st.synthetic_detect(close, prox)["regime_p"] < 0.05:
            fires += 1
    assert fires <= 2                                         # ~5% false-positive budget

    close, prox = data.synthetic_world(amp=0.02, seed=737)
    planted = st.synthetic_detect(close, prox)
    assert planted["regime_p"] < 0.01                         # a real cycle is detected
    assert abs(planted["t_cos"]) > 4


def test_phase_regression_null_is_flat():
    """On the amp=0 synthetic world the 11-year wave explains ~nothing."""
    close, prox = data.synthetic_world(amp=0.0, seed=737)
    ar = st.abnormal_returns(st.monthly_returns(close))
    pr = st.phase_regression(ar, prox)
    assert pr["r2"] < 0.02                                    # < 2% variance explained under the null
