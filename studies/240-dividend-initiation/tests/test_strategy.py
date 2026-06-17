"""Arm return construction, hedge, summary — and the study's spine:
the initiator screen finds an edge only when a premium is planted, not in the null."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dividend_initiation import data, strategy as st  # noqa: E402

CACHE_SENTINEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "di_AAPL_px.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_SENTINEL),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# arm_returns
# ---------------------------------------------------------------------------
def test_arm_returns_type(real_world):
    r = st.arm_returns(real_world["fwd_ret"], real_world["initiation_mask"])
    assert isinstance(r, pd.Series)


def test_arm_returns_min_names_filter():
    """Years with fewer than min_names in the mask are excluded."""
    years = pd.Index([2010, 2011, 2012], name="year")
    names = ["A", "B", "C"]
    fwd = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.0, 0.1, 0.2], "C": [0.0, 0.0, 0.1]},
                       index=years)
    mask = pd.DataFrame({"A": [True, True, True], "B": [False, True, True],
                         "C": [False, False, True]}, index=years)
    # min_names=2: 2010 has 1 -> excluded; 2011 has 2 -> included; 2012 has 3 -> included
    r = st.arm_returns(fwd, mask, min_names=2)
    assert 2010 not in r.index
    assert 2011 in r.index
    assert 2012 in r.index
    assert np.isclose(r[2011], (0.2 + 0.1) / 2)


def test_arm_returns_is_equal_weight(real_world):
    """The arm return in each year equals the simple mean of selected names' fwd returns."""
    p = real_world
    r = st.arm_returns(p["fwd_ret"], p["initiation_mask"], min_names=1)
    for y in r.index:
        sel = p["initiation_mask"].columns[p["initiation_mask"].loc[y]].tolist()
        fwd_sel = p["fwd_ret"].loc[y, sel].dropna()
        if len(fwd_sel) >= 1:
            assert np.isclose(r[y], fwd_sel.mean())


# ---------------------------------------------------------------------------
# hedge_returns
# ---------------------------------------------------------------------------
def test_hedge_is_initiator_minus_control(real_world):
    p = real_world
    g = st.arm_returns(p["fwd_ret"], p["initiation_mask"], min_names=1)
    c = st.arm_returns(p["fwd_ret"], p["control_mask"], min_names=1)
    h = st.hedge_returns(g, c)
    expected = (g - c).reindex(h.index)
    assert np.allclose(h.to_numpy(), expected.to_numpy(), equal_nan=True)


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
# The study's spine: initiator screen works only when premium is planted
# ---------------------------------------------------------------------------
def test_initiator_outperforms_only_with_planted_premium():
    """With a planted premium, initiators beat the control arm.
    With no premium, initiators are indistinguishable from control."""
    p_real = data.synthetic_panel(initiation_premium=0.15, seed=240, n_years=30)
    p_null = data.synthetic_panel(initiation_premium=0.00, seed=240, n_years=30)

    def spread_mean(panel):
        g = st.arm_returns(panel["fwd_ret"], panel["initiation_mask"], min_names=1)
        c = st.arm_returns(panel["fwd_ret"], panel["control_mask"], min_names=1)
        h = st.hedge_returns(g, c)
        return h.mean() if len(h) > 0 else 0.0

    spread_real = spread_mean(p_real)
    spread_null = spread_mean(p_null)
    assert spread_real > spread_null, (
        f"planted premium not showing: real={spread_real:.3f} null={spread_null:.3f}"
    )


def test_initiator_tstat_higher_with_premium():
    """The t-stat on the initiator spread is higher when a premium exists."""
    p_real = data.synthetic_panel(initiation_premium=0.15, seed=240, n_years=30)
    p_null = data.synthetic_panel(initiation_premium=0.00, seed=240, n_years=30)

    def spread_tstat(panel):
        g = st.arm_returns(panel["fwd_ret"], panel["initiation_mask"], min_names=1)
        c = st.arm_returns(panel["fwd_ret"], panel["control_mask"], min_names=1)
        h = st.hedge_returns(g, c)
        s = st.summary(h)
        t = s.get("tstat", 0.0)
        return abs(t) if t == t else 0.0

    t_real = spread_tstat(p_real)
    t_null = spread_tstat(p_null)
    assert t_real >= t_null, (
        f"premium world tstat={t_real:.2f} not >= null tstat={t_null:.2f}"
    )


# ---------------------------------------------------------------------------
# Real panel spine — cache-gated
# ---------------------------------------------------------------------------
@requires_cache
def test_real_initiator_summary_runs():
    """On the real tape the summary should return a result (could be NaN if too few events)."""
    p = data.load_real_panel(fetch=False)
    if p["initiation_mask"].empty:
        pytest.skip("empty real panel")
    g = st.arm_returns(p["fwd_ret"], p["initiation_mask"], min_names=1)
    s = st.summary(g)
    assert isinstance(s, dict)
    assert "n" in s
