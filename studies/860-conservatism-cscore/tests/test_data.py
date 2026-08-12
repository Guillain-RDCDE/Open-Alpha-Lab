"""Data-layer invariants: the C-score construction (reserve sum, scaling, NOA, no look-ahead)
and the synthetic panel's shape — all offline, fixed seeds. A real-cache smoke test is gated on
the cache existing (absent on CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from conservatism import data  # noqa: E402


def _instant(vals, start="2018-03-31"):
    ends = pd.date_range(start, periods=len(vals), freq="QE")
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_build_signal_reserve_sum_and_scaling():
    ada = _instant([10, 12, 14])
    invres = _instant([20, 20, 20])
    dtva = _instant([5, 5, 5])
    assets = _instant([1000, 1000, 1000])
    cash = _instant([100, 100, 100])
    liab = _instant([400, 400, 400])
    debt = _instant([150, 150, 150])
    ni = _instant([50, 50, 50])
    sig = data.build_signal(ada, invres, dtva, assets, cash, liab, debt, ni)
    r0 = sig.iloc[0]
    # reserves = 10 + 20 + 5 = 35 ; cscore = 35/1000 = 0.035
    assert r0["reserves"] == pytest.approx(35.0, abs=1e-9)
    assert r0["cscore"] == pytest.approx(0.035, abs=1e-9)
    # NOA = 1000 - 100 - (400 - 150) = 650 ; cscore_noa = 35/650
    assert r0["noa"] == pytest.approx(650.0, abs=1e-9)
    assert r0["cscore_noa"] == pytest.approx(35.0 / 650.0, abs=1e-9)


def test_build_signal_partial_reserves_still_counts():
    # only the allowance for doubtful accounts is tagged; the row must still form (a floor)
    ada = _instant([8, 9, 10])
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    assets = _instant([500, 500, 500])
    sig = data.build_signal(ada, empty, empty, assets, empty, empty, empty, empty)
    assert len(sig) == 3
    assert sig.iloc[0]["reserves"] == pytest.approx(8.0, abs=1e-9)
    # no cash/liab/debt -> NOA is NaN, cscore_noa NaN, but cscore is fine
    assert np.isnan(sig.iloc[0]["noa"])
    assert sig.iloc[0]["cscore"] == pytest.approx(8.0 / 500.0, abs=1e-9)


def test_build_signal_no_lookahead_filed_after_end():
    ada = _instant([10, 11, 12])
    assets = _instant([1000, 1000, 1000])
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    sig = data.build_signal(ada, empty, empty, assets, empty, empty, empty, empty)
    # the signal for period end E must be stamped with the FILING date (E + ~35d), not E
    assert (sig["filed"] > sig["end"]).all()


def test_build_signal_drops_nonpositive_assets():
    ada = _instant([10, 11, 12])
    assets = _instant([0, 1000, 1000])
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    sig = data.build_signal(ada, empty, empty, assets, empty, empty, empty, empty)
    # the first quarter has Assets = 0 -> dropped; only 2 rows remain
    assert len(sig) == 2


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "cscore"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["cscore"].to_numpy(), e2["cscore"].to_numpy())


@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_schema():
    px, ev = data.load_real()
    for c in ("ticker", "end", "filed", "reserves", "assets", "cscore"):
        assert c in ev.columns
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
    assert (ev["cscore"] >= 0).mean() > 0.9      # reserve ratios are non-negative
    assert px.index.max() <= pd.Timestamp(data.AS_OF)
