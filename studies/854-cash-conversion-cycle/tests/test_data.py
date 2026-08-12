"""Data-layer invariants: the CCC construction (the three days-ratios, CCC = DSO + DIO − DPO,
no look-ahead, the signed score) and the synthetic panel's shape — all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ccc_signal import data  # noqa: E402

CACHE = [data.PRICES_CACHE, data.EVENTS_CACHE]


def _instant_series(vals, start="2018-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_days_ratio_math():
    # level = 100, one quarter of flow = 200 -> days = 100 * (365/4) / 200 = 45.625
    assert data._days(100.0, 200.0) == pytest.approx(100.0 * (365.0 / 4.0) / 200.0, abs=1e-9)


def test_days_rejects_nonpositive_flow():
    assert data._days(100.0, 0.0) is None
    assert data._days(100.0, -5.0) is None


def test_ccc_identity():
    # CCC = DSO + DIO - DPO
    assert data._ccc(40.0, 60.0, 30.0) == pytest.approx(40.0 + 60.0 - 30.0)
    assert data._ccc(None, 60.0, 30.0) is None
    assert data._ccc(40.0, None, 30.0) is None
    assert data._ccc(40.0, 60.0, None) is None


def test_build_signal_ccc_and_sign():
    # Inventory JUMPS from 100 -> 140 YoY at q5 (idx4), everything else flat.
    # DIO rises -> CCC rises -> positive change -> UNATTRACTIVE (negative score).
    ar = _instant_series([100, 100, 100, 100, 100])
    inv = _instant_series([100, 100, 100, 100, 140])
    ap = _instant_series([80, 80, 80, 80, 80])
    rev = _instant_series([200, 200, 200, 200, 200])
    cogs = _instant_series([120, 120, 120, 120, 120])
    sig = data.build_signal(ar, inv, ap, rev, cogs)
    row = sig.iloc[0]
    dq = 365.0 / 4.0
    dso = 100.0 * dq / 200.0
    dio_now = 140.0 * dq / 120.0
    dio_old = 100.0 * dq / 120.0
    dpo = 80.0 * dq / 120.0
    ccc_now = dso + dio_now - dpo
    ccc_old = dso + dio_old - dpo
    assert row["ccc"] == pytest.approx(ccc_now, abs=1e-9)
    assert row["ccc_yoy_chg"] == pytest.approx(ccc_now - ccc_old, abs=1e-9)
    assert row["ccc_yoy_chg"] > 0                        # inventory piled up -> CCC rose
    # the score is the NEGATED change (high score = falling CCC = long)
    assert row["ccc_score"] == pytest.approx(-row["ccc_yoy_chg"], abs=1e-12)


def test_build_signal_payables_stretch_shortens_ccc():
    # Payables STRETCH from 80 -> 140 YoY -> DPO rises -> CCC FALLS -> negative change,
    # POSITIVE score (attractive).
    ar = _instant_series([100, 100, 100, 100, 100])
    inv = _instant_series([100, 100, 100, 100, 100])
    ap = _instant_series([80, 80, 80, 80, 140])
    rev = _instant_series([200, 200, 200, 200, 200])
    cogs = _instant_series([120, 120, 120, 120, 120])
    sig = data.build_signal(ar, inv, ap, rev, cogs)
    row = sig.iloc[0]
    assert row["ccc_yoy_chg"] < 0
    assert row["ccc_score"] > 0


def test_build_signal_no_lookahead_filed_date():
    ar = _instant_series([100, 100, 100, 100, 120])
    inv = _instant_series([100] * 5)
    ap = _instant_series([80] * 5)
    rev = _instant_series([200] * 5)
    cogs = _instant_series([120] * 5)
    sig = data.build_signal(ar, inv, ap, rev, cogs)
    r = sig.iloc[0]
    assert r["filed"] > r["end"]                          # stamped with the FILING date


def test_build_signal_drops_missing_prior():
    # only one year of data -> no YoY match possible -> empty
    ar = _instant_series([100, 100, 100])
    inv = _instant_series([100, 100, 100])
    ap = _instant_series([80, 80, 80])
    rev = _instant_series([200, 200, 200])
    cogs = _instant_series([120, 120, 120])
    sig = data.build_signal(ar, inv, ap, rev, cogs)
    assert len(sig) == 0


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "ccc_yoy_chg", "ccc_score"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50
    # score is exactly the negated change
    assert np.allclose(ev["ccc_score"].to_numpy(), -ev["ccc_yoy_chg"].to_numpy())


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["ccc_yoy_chg"].to_numpy(), e2["ccc_yoy_chg"].to_numpy())


@pytest.mark.skipif(not all(os.path.exists(p) for p in CACHE),
                    reason="real cache absent offline CI")
def test_real_cache_sane():
    px, ev = data.load_real()
    assert px.shape[0] > 100 and px.shape[1] >= 10
    assert {"ticker", "filed", "ccc_yoy_chg", "ccc_score"}.issubset(ev.columns)
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
