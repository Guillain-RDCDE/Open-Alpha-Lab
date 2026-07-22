"""Data-layer invariants: the signal construction (YoY math, no look-ahead, scaling) and the
synthetic panel's shape — all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from deferred_revenue import data  # noqa: E402


def _quarter_series(vals, start="2018-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_build_signal_yoy_math():
    # deferred revenue grows 10% each quarter YoY: 100,110,121,133 -> at q5 vs q1 = +10%
    defrev = _quarter_series([100, 100, 100, 100, 110, 121, 133.1, 146.41])
    rev = _quarter_series([50] * 8)
    assets = _quarter_series([1000] * 8)
    sig = data.build_signal(defrev, rev, assets)
    # first row with a YoY match is the 5th quarter (index 4): 110/100 - 1 = 0.10
    row = sig.iloc[0]
    assert row["defrev_yoy"] == pytest.approx(0.10, abs=1e-9)
    # scaled: (110-100)/1000 = 0.01
    assert row["defrev_chg_assets"] == pytest.approx(0.01, abs=1e-9)


def test_build_signal_no_lookahead_filed_date():
    defrev = _quarter_series([100, 100, 100, 100, 120])
    rev = _quarter_series([50] * 5)
    assets = _quarter_series([1000] * 5)
    sig = data.build_signal(defrev, rev, assets)
    # the signal for period end E must be stamped with the FILING date (E + ~35d), not E
    r = sig.iloc[0]
    assert r["filed"] > r["end"]


def test_build_signal_drops_nonpositive_prior():
    defrev = _quarter_series([0, 100, 100, 100, 120])
    rev = _quarter_series([50] * 5)
    assets = _quarter_series([1000] * 5)
    sig = data.build_signal(defrev, rev, assets)
    # the q5 (idx4) YoY denominator is the q1 value 0 -> that row must be dropped
    assert (sig["end"] == defrev["end"].iloc[4]).sum() == 0


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "defrev_yoy"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["defrev_yoy"].to_numpy(), e2["defrev_yoy"].to_numpy())
