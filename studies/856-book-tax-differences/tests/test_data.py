"""Data-layer invariants: the book-tax-difference construction (gross-up math, statutory-rate
switch, no look-ahead, scaling) and the synthetic panel's shape — all offline, fixed seeds.

A real-cache smoke test is gated on the cache existing (absent on CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from book_tax_diff import data  # noqa: E402


def _year_series(vals, start="2010-12-31"):
    ends = pd.date_range(start, periods=len(vals), freq="YE")
    filed = ends + pd.Timedelta(days=55)
    return pd.DataFrame({"end": ends, "filed": filed, "val": [float(v) for v in vals]})


def test_statutory_rate_switch():
    assert data.statutory_rate(pd.Timestamp("2016-12-31")) == pytest.approx(0.35)
    assert data.statutory_rate(pd.Timestamp("2017-12-31")) == pytest.approx(0.35)
    assert data.statutory_rate(pd.Timestamp("2018-12-31")) == pytest.approx(0.21)
    assert data.statutory_rate(pd.Timestamp("2024-12-31")) == pytest.approx(0.21)


def test_build_signal_btd_math_pre_tcja():
    # 2010-2013 fiscal years: statutory 35%. pretax=1000, tax=210 -> implied taxable = 210/.35=600
    # BTD = 1000 - 600 = 400; scaled by assets 4000 -> btd_assets = 0.10
    pretax = _year_series([1000, 1000, 1000, 1000], start="2010-12-31")
    tax = _year_series([210, 210, 210, 210], start="2010-12-31")
    assets = _year_series([4000, 4000, 4000, 4000], start="2010-12-31")
    sig = data.build_signal(pretax, tax, assets)
    r = sig.iloc[0]
    assert r["btd"] == pytest.approx(400.0, abs=1e-6)
    assert r["btd_assets"] == pytest.approx(0.10, abs=1e-9)
    assert r["btd_neg"] == pytest.approx(-0.10, abs=1e-9)
    assert r["roa"] == pytest.approx(0.25, abs=1e-9)


def test_build_signal_uses_post_tcja_rate():
    # a 2019 fiscal year uses the 21% statutory rate. pretax=1000, tax=210 -> implied=1000 -> BTD=0
    pretax = _year_series([1000], start="2019-12-31")
    tax = _year_series([210], start="2019-12-31")
    assets = _year_series([5000], start="2019-12-31")
    sig = data.build_signal(pretax, tax, assets)
    assert sig.iloc[0]["btd"] == pytest.approx(0.0, abs=1e-6)
    assert sig.iloc[0]["btd_assets"] == pytest.approx(0.0, abs=1e-9)


def test_build_signal_no_lookahead_filed_date():
    pretax = _year_series([1000, 1100], start="2012-12-31")
    tax = _year_series([300, 330], start="2012-12-31")
    assets = _year_series([5000, 5000], start="2012-12-31")
    sig = data.build_signal(pretax, tax, assets)
    r = sig.iloc[0]
    # the signal for fiscal year end E must be stamped with the FILING date (E + ~55d), not E
    assert r["filed"] > r["end"]


def test_build_signal_change_and_next_roa():
    # rising book-tax gap year over year, plus a next-year ROA link
    pretax = _year_series([1000, 1200, 1400], start="2013-12-31")
    tax = _year_series([350, 350, 350], start="2013-12-31")   # tax flat -> implied taxable flat
    assets = _year_series([10000, 10000, 10000], start="2013-12-31")
    sig = data.build_signal(pretax, tax, assets)
    # year 2 (idx1) change in btd_assets should be positive (gap widened)
    assert sig.iloc[1]["d_btd_assets"] > 0
    # roa_next of the first row equals second row's roa (1200/10000 = 0.12)
    assert sig.iloc[0]["roa_next"] == pytest.approx(0.12, abs=1e-9)


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_years=14, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "btd_assets", "btd_neg"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert len(ev) > 50
    # btd_neg is exactly the negation of btd_assets
    assert np.allclose(ev["btd_neg"].to_numpy(), -ev["btd_assets"].to_numpy())


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_years=12, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_years=12, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["btd_assets"].to_numpy(), e2["btd_assets"].to_numpy())


def test_synthetic_index_no_timestamp_overflow():
    # guard against the pandas nanosecond-Timestamp horizon trap on long panels
    prices, ev = data.synthetic_panel(n_names=6, n_years=16, edge=0.0, seed=3)
    assert prices.index.max() < pd.Timestamp("2100-01-01")


@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_smoke():
    px, ev = data.load_real()
    assert px.shape[1] > 5 and len(ev) > 50
    assert {"ticker", "filed", "btd_assets", "btd_neg", "roa", "roa_next"}.issubset(ev.columns)
    assert (ev["filed"] <= pd.Timestamp(data.AS_OF)).all()
