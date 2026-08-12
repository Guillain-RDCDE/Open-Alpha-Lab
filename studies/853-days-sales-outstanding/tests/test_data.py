"""Data-layer invariants: the DSO signal construction (the days math, no look-ahead, the signed
score) and the synthetic panel's shape — all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dso_signal import data  # noqa: E402

CACHE = [data.PRICES_CACHE, data.EVENTS_CACHE]


def _instant_series(vals, start="2018-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_dso_level_math():
    # AR = 100, one quarter of revenue = 200 -> DSO = 100 * (365/4) / 200 = 45.625 days
    assert data._dso(100.0, 200.0) == pytest.approx(100.0 * (365.0 / 4.0) / 200.0, abs=1e-9)


def test_dso_rejects_nonpositive_revenue():
    assert data._dso(100.0, 0.0) is None
    assert data._dso(100.0, -5.0) is None


def test_build_signal_dso_change_and_sign():
    # AR flat at 100; revenue DROPS from 200 -> 160 YoY at q5 (idx4). DSO rises -> positive buildup
    ar = _instant_series([100, 100, 100, 100, 100])
    rev = _instant_series([200, 200, 200, 200, 160])
    sig = data.build_signal(ar, rev)
    row = sig.iloc[0]
    dso_now = 100.0 * (365.0 / 4.0) / 160.0
    dso_prior = 100.0 * (365.0 / 4.0) / 200.0
    assert row["dso"] == pytest.approx(dso_now, abs=1e-9)
    assert row["dso_yoy_chg"] == pytest.approx(dso_now - dso_prior, abs=1e-9)
    assert row["dso_yoy_chg"] > 0                       # receivables grew faster than sales
    # the score is the NEGATED change (high score = low buildup = long)
    assert row["dso_score"] == pytest.approx(-row["dso_yoy_chg"], abs=1e-12)


def test_build_signal_no_lookahead_filed_date():
    ar = _instant_series([100, 100, 100, 100, 120])
    rev = _instant_series([200] * 5)
    sig = data.build_signal(ar, rev)
    r = sig.iloc[0]
    assert r["filed"] > r["end"]                        # stamped with the FILING date, not period end


def test_build_signal_drops_missing_prior():
    # only one year of data -> no YoY match possible -> empty
    ar = _instant_series([100, 100, 100])
    rev = _instant_series([200, 200, 200])
    sig = data.build_signal(ar, rev)
    assert len(sig) == 0


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "dso_yoy_chg", "dso_score"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50
    # score is exactly the negated change
    assert np.allclose(ev["dso_score"].to_numpy(), -ev["dso_yoy_chg"].to_numpy())


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["dso_yoy_chg"].to_numpy(), e2["dso_yoy_chg"].to_numpy())


@pytest.mark.skipif(not all(os.path.exists(p) for p in CACHE),
                    reason="real cache absent offline CI")
def test_real_cache_sane():
    px, ev = data.load_real()
    assert px.shape[0] > 100 and px.shape[1] >= 10
    assert {"ticker", "filed", "dso_yoy_chg", "dso_score"}.issubset(ev.columns)
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
