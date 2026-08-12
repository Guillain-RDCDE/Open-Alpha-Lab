"""Data-layer invariants: the ROIC construction (identity math, TTM, no look-ahead), the
tax-rate invariance of the sort, and the synthetic panel's shape — all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roic_premium import data  # noqa: E402

_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def _q_series(vals, start="2015-03-31"):
    ends = pd.period_range(start, periods=len(vals), freq="Q").to_timestamp(how="end").normalize()
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_ttm_sums_trailing_four_quarters():
    q = _q_series([10, 20, 30, 40, 50, 60])
    t = data.ttm(q)
    # first TTM row is the 4th quarter end: 10+20+30+40 = 100
    assert t.iloc[0]["val"] == pytest.approx(100.0)
    # next: 20+30+40+50 = 140 ; then 30+40+50+60 = 180
    assert t.iloc[1]["val"] == pytest.approx(140.0)
    assert t.iloc[2]["val"] == pytest.approx(180.0)


def test_ttm_drops_partial_year():
    q = _q_series([10, 20, 30])   # only 3 quarters -> no full TTM
    assert data.ttm(q).empty


def test_build_signal_roic_identity():
    # one balance-sheet quarter: OIL_ttm = 400, equity = 1000, debt = 500, cash = 100
    # invested capital = 500 + 1000 - 100 = 1400 ; NOPAT = 400*(1-0.21) = 316
    # ROIC = 316 / 1400
    oil = _q_series([100, 100, 100, 100])       # ttm -> 400 at the 4th end
    oil_ttm = data.ttm(oil)
    E = oil_ttm.iloc[0]["end"]
    equity = pd.DataFrame({"end": [E], "filed": [E + pd.Timedelta(days=35)], "val": [1000.0]})
    debt = pd.DataFrame({"end": [E], "filed": [E], "val": [500.0]})
    cash = pd.DataFrame({"end": [E], "filed": [E], "val": [100.0]})
    ni_ttm = data.ttm(_q_series([50, 50, 50, 50]))
    assets = pd.DataFrame({"end": [E], "filed": [E], "val": [3000.0]})
    gp_ttm = data.ttm(_q_series([200, 200, 200, 200]))
    sig = data.build_signal(oil_ttm, ni_ttm, gp_ttm, equity, debt, cash, assets)
    row = sig.iloc[0]
    assert row["invested_capital"] == pytest.approx(1400.0)
    assert row["roic"] == pytest.approx(316.0 / 1400.0, rel=1e-9)
    assert row["roe"] == pytest.approx(200.0 / 1000.0, rel=1e-9)      # NI_ttm 200 / equity 1000
    assert row["gp"] == pytest.approx(800.0 / 3000.0, rel=1e-9)       # GP_ttm 800 / assets 3000


def test_build_signal_tax_rate_invariant_ranking():
    # A flat tax rate is a common scalar -> it must NOT change the cross-sectional ROIC ranking.
    oil = data.ttm(_q_series([100, 100, 100, 100]))
    E = oil.iloc[0]["end"]

    def _one(ic_equity, tax):
        equity = pd.DataFrame({"end": [E], "filed": [E], "val": [float(ic_equity)]})
        debt = pd.DataFrame({"end": [E], "filed": [E], "val": [0.0]})
        cash = pd.DataFrame({"end": [E], "filed": [E], "val": [0.0]})
        empty = pd.DataFrame(columns=["end", "filed", "val"])
        return data.build_signal(oil, empty, empty, equity, debt, cash, empty,
                                 tax_rate=tax).iloc[0]["roic"]

    a21, b21 = _one(1000, 0.21), _one(2000, 0.21)
    a35, b35 = _one(1000, 0.35), _one(2000, 0.35)
    # same ordering under both tax rates
    assert (a21 > b21) == (a35 > b35)
    # and the ratio between two names is identical across tax rates (common scalar cancels)
    assert (a21 / b21) == pytest.approx(a35 / b35, rel=1e-12)


def test_build_signal_drops_nonpositive_invested_capital():
    oil = data.ttm(_q_series([100, 100, 100, 100]))
    E = oil.iloc[0]["end"]
    equity = pd.DataFrame({"end": [E], "filed": [E], "val": [100.0]})
    debt = pd.DataFrame({"end": [E], "filed": [E], "val": [0.0]})
    cash = pd.DataFrame({"end": [E], "filed": [E], "val": [500.0]})   # cash >> debt+equity
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    sig = data.build_signal(oil, empty, empty, equity, debt, cash, empty)
    assert sig.empty                                                  # IC = 0+100-500 < 0 -> dropped


def test_synthetic_panel_shapes_and_no_lookahead():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=24, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "roic", "roic_chg", "roe", "gp"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()          # signal stamped at filing, after period end
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["roic"].to_numpy(), e2["roic"].to_numpy())


def test_fingerprint_stable():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    assert data.fingerprint(df) == data.fingerprint(df.copy())
    assert len(data.fingerprint(df)) == 12


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped when the (git-ignored) cache is absent on CI
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_wellformed():
    px, ev = data.load_real()
    assert px.shape[1] >= 10 and len(px) > 200
    for c in ("ticker", "end", "filed", "invested_capital", "roic", "roic_chg", "roe", "gp"):
        assert c in ev.columns
    assert (ev["filed"] > ev["end"]).all()
    assert ev["filed"].max() <= pd.Timestamp(data.AS_OF)
    assert (ev["invested_capital"] > 0).all()
