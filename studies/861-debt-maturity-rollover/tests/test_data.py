"""Data-layer invariants: the short-term-share construction (ratio math, no look-ahead, missing-leg
handling, degenerate drops) and the synthetic panel's shape — all offline, fixed seeds.

The one test that touches the real cache is skipped when it is absent (offline CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from debt_maturity import data  # noqa: E402


def _quarter_series(vals, start="2018-03-31"):
    ends = pd.period_range(start, periods=len(vals), freq="Q").to_timestamp(how="end").normalize()
    filed = ends + pd.Timedelta(days=35)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_build_signal_share_math():
    # DebtCurrent=20, LongTermDebtCurrent=30, LongTermDebtNoncurrent=50 -> share = 50/100 = 0.5
    dc = _quarter_series([20, 20, 20])
    lc = _quarter_series([30, 30, 30])
    nc = _quarter_series([50, 50, 50])
    assets = _quarter_series([1000, 1000, 1000])
    sig = data.build_signal(dc, lc, nc, assets)
    assert len(sig) == 3
    assert sig["st_share"].iloc[0] == pytest.approx(0.5, abs=1e-9)
    # scaled: (20+30)/1000 = 0.05
    assert sig["st_debt_assets"].iloc[0] == pytest.approx(0.05, abs=1e-9)
    assert sig["total_debt"].iloc[0] == pytest.approx(100.0, abs=1e-9)


def test_build_signal_missing_short_leg_is_zero():
    # no DebtCurrent at all -> share = LongTermDebtCurrent / (LC + NC)
    dc = _quarter_series([])  # empty
    lc = _quarter_series([10, 10])
    nc = _quarter_series([90, 90])
    assets = _quarter_series([1000, 1000])
    sig = data.build_signal(dc, lc, nc, assets)
    assert sig["st_share"].iloc[0] == pytest.approx(0.10, abs=1e-9)


def test_build_signal_no_lookahead_filed_date():
    dc = _quarter_series([20, 20])
    lc = _quarter_series([30, 30])
    nc = _quarter_series([50, 50])
    assets = _quarter_series([1000, 1000])
    sig = data.build_signal(dc, lc, nc, assets)
    # the signal for period end E must be stamped with the FILING date (E + ~35d), not E
    assert (sig["filed"] > sig["end"]).all()


def test_build_signal_drops_zero_total():
    # a quarter with no debt at all (all legs zero) must be dropped
    dc = _quarter_series([0, 20])
    lc = _quarter_series([0, 30])
    nc = _quarter_series([0, 50])
    assets = _quarter_series([1000, 1000])
    sig = data.build_signal(dc, lc, nc, assets)
    # first quarter total=0 dropped; only the second survives
    assert len(sig) == 1
    assert sig["st_share"].iloc[0] == pytest.approx(0.5, abs=1e-9)


def test_build_signal_share_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 12
    dc = _quarter_series(rng.uniform(0, 50, n))
    lc = _quarter_series(rng.uniform(0, 50, n))
    nc = _quarter_series(rng.uniform(1, 200, n))
    assets = _quarter_series(rng.uniform(500, 2000, n))
    sig = data.build_signal(dc, lc, nc, assets)
    assert (sig["st_share"] >= 0).all() and (sig["st_share"] <= 1).all()


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=20, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "st_share"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert (ev["st_share"] >= 0).all() and (ev["st_share"] <= 1).all()
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=15, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["st_share"].to_numpy(), e2["st_share"].to_numpy())


@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_shape_if_present():
    px, ev = data.load_real()
    assert px.shape[0] > 100 and px.shape[1] > 5
    assert {"ticker", "st_share", "filed"}.issubset(ev.columns)
    assert (ev["st_share"] >= 0).all() and (ev["st_share"] <= 1).all()
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
