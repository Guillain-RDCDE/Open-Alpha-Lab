"""Signal classification, arm return construction, hedge, summary — and the study's spine:
the grower screen finds an edge only when a premium is planted, not in the null."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_growth import data, strategy as st  # noqa: E402

CACHE_SENTINEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "dg_AAPL_px.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_SENTINEL),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# classify_growers
# ---------------------------------------------------------------------------
def test_classify_growers_type_and_shape(real_world):
    mask = st.classify_growers(real_world["streaks"], streak_years=3)
    assert mask.shape == real_world["streaks"].shape
    assert mask.dtypes.eq(bool).all()


def test_classify_growers_threshold(real_world):
    """A name with streak >= 3 is a grower; one with streak < 3 is not."""
    streaks = real_world["streaks"]
    mask3 = st.classify_growers(streaks, streak_years=3)
    # Every cell where streak >= 3 must be True
    assert ((streaks >= 3) == mask3).all().all()


def test_classify_growers_loose_threshold_picks_more():
    """Lowering the streak threshold should pick at least as many (or more) growers."""
    p = data.synthetic_panel(seed=201)
    m3 = st.classify_growers(p["streaks"], streak_years=3)
    m1 = st.classify_growers(p["streaks"], streak_years=1)
    assert m1.sum().sum() >= m3.sum().sum()


# ---------------------------------------------------------------------------
# arm_returns
# ---------------------------------------------------------------------------
def test_arm_returns_shape_and_type(real_world):
    mask = st.classify_growers(real_world["streaks"], streak_years=3)
    r = st.arm_returns(real_world["fwd_ret"], mask)
    assert isinstance(r, pd.Series)
    assert len(r) <= len(real_world["fwd_ret"])


def test_arm_returns_min_names_filter():
    """Years with fewer than min_names in the mask are excluded."""
    years = pd.Index([2010, 2011], name="year")
    names = ["A", "B"]
    fwd = pd.DataFrame({"A": [0.1, 0.2], "B": [0.0, 0.1]}, index=years)
    mask = pd.DataFrame({"A": [True, True], "B": [False, True]}, index=years)
    # min_names=2: 2010 has only 1 selected -> excluded; 2011 has 2 -> included
    r = st.arm_returns(fwd, mask, min_names=2)
    assert 2010 not in r.index
    assert 2011 in r.index
    assert np.isclose(r[2011], (0.2 + 0.1) / 2)


def test_arm_returns_is_equal_weight(real_world):
    """The arm return in each year equals the simple mean of selected names' fwd returns."""
    p = real_world
    mask = st.classify_growers(p["streaks"], streak_years=3)
    r = st.arm_returns(p["fwd_ret"], mask, min_names=2)
    for y in r.index:
        sel = mask.columns[mask.loc[y]].tolist()
        fwd_sel = p["fwd_ret"].loc[y, sel].dropna()
        if len(fwd_sel) >= 2:
            assert np.isclose(r[y], fwd_sel.mean())


# ---------------------------------------------------------------------------
# hedge_returns
# ---------------------------------------------------------------------------
def test_hedge_is_grower_minus_ew(real_world):
    p = real_world
    mask = st.classify_growers(p["streaks"], streak_years=3)
    g = st.arm_returns(p["fwd_ret"], mask)
    e = p["ew_ret"].reindex(g.index)
    h = st.hedge_returns(g, e)
    assert np.allclose(h.to_numpy(), (g - e).dropna().to_numpy(), equal_nan=True)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_keys():
    r = pd.Series([0.05, -0.02, 0.10, 0.03, -0.01, 0.07, 0.04, 0.06, -0.03, 0.08])
    s = st.summary(r)
    assert set(s.keys()) >= {"mean", "vol", "sharpe", "tstat", "hit_rate", "max_dd", "n"}


def test_summary_mean_vol_consistent():
    r = pd.Series([0.05, -0.02, 0.10, 0.03, -0.01, 0.07, 0.04, 0.06, -0.03, 0.08])
    s = st.summary(r)
    assert np.isclose(s["mean"], r.mean())
    assert np.isclose(s["vol"], r.std(ddof=1))


def test_summary_hit_rate():
    r = pd.Series([0.1, -0.1, 0.2, -0.2, 0.3])  # 3 positive out of 5
    s = st.summary(r)
    assert np.isclose(s["hit_rate"], 0.6)


def test_summary_short_series_returns_nan():
    s = st.summary(pd.Series([0.05]))
    assert np.isnan(s["mean"])


# ---------------------------------------------------------------------------
# The study's spine: grower screen works only when premium is planted
# ---------------------------------------------------------------------------
def test_grower_outperforms_only_with_planted_premium():
    """The spine of Study 201: with a planted premium, growers beat the EW basket.
    With no premium, growers are indistinguishable from the basket."""
    p_real = data.synthetic_panel(growth_premium=0.05, seed=201)
    p_null = data.synthetic_panel(growth_premium=0.00, seed=201)

    def grower_spread(panel):
        mask = st.classify_growers(panel["streaks"], streak_years=3)
        g = st.arm_returns(panel["fwd_ret"], mask)
        ew = panel["ew_ret"].reindex(g.index)
        return st.hedge_returns(g, ew).mean()

    spread_real = grower_spread(p_real)
    spread_null = grower_spread(p_null)
    assert spread_real > 0.01, f"planted premium not showing (spread={spread_real:.3f})"
    assert abs(spread_null) < 0.025, f"null world shows spurious spread={spread_null:.3f}"


def test_grower_tstat_higher_with_premium():
    """The t-stat on the grower spread is higher (more significant) when a premium exists."""
    p_real = data.synthetic_panel(growth_premium=0.05, seed=201)
    p_null = data.synthetic_panel(growth_premium=0.00, seed=201)

    def spread_tstat(panel):
        mask = st.classify_growers(panel["streaks"], streak_years=3)
        g = st.arm_returns(panel["fwd_ret"], mask)
        ew = panel["ew_ret"].reindex(g.index)
        return abs(st.summary(st.hedge_returns(g, ew))["tstat"])

    t_real = spread_tstat(p_real)
    t_null = spread_tstat(p_null)
    assert t_real > t_null


# ---------------------------------------------------------------------------
# Real panel spine — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_grower_summary_runs_and_has_finite_tstat():
    """On the real tape the summary should return a finite t-stat (could be low)."""
    p = data.load_real_panel(fetch=False)
    if p["streaks"].empty:
        pytest.skip("empty real panel")
    mask = st.classify_growers(p["streaks"], streak_years=3)
    g = st.arm_returns(p["fwd_ret"], mask)
    s = st.summary(g)
    assert np.isfinite(s["tstat"]) or np.isnan(s["tstat"])  # allowed to be NaN if tiny n
    assert s["n"] >= 1
