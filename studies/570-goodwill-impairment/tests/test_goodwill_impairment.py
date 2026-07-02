"""Tests for Study 570 (Goodwill-Impairment) — deterministic, offline, no network.

They pin the engine's two required behaviours: it *recovers* a planted overpaid-acquisition effect
(a positive low-minus-high spread, a negative firm slope, a higher high-goodwill impairment rate)
and it stays *flat at the null* — the two halves of an honest positive control.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from goodwill_impairment import data, strategy as st  # noqa: E402


def test_synthetic_panel_is_deterministic():
    p1, t1 = data.synthetic_panel(seed=570)
    p2, t2 = data.synthetic_panel(seed=570)
    assert data.fingerprint(p1) == data.fingerprint(p2)
    assert t1 == t2
    # schema the strategy consumes
    assert set(["goodwill_ta", "impaired", "forward_ret"]).issubset(p1.columns)
    assert p1["impaired"].isin([0.0, 1.0]).all()


def test_effect_recovered_when_planted():
    panel, truth = data.synthetic_panel(ret_alpha=-0.14, seed=570)
    assert truth.has_effect
    b = st.bucket_returns(panel, 0.2)
    ls = st.long_short_tstat(b)
    reg = st.cross_section_regression(panel)
    # low-goodwill beats high-goodwill (positive spread) with a strong t
    assert b["spread"] > 0
    assert ls["t"] > 2.0
    # firm slope is negative (the puzzle) and significant
    assert reg["slope"] < 0
    assert reg["slope_t"] < -2.0
    # placebo puts the spread in the tail
    assert st.placebo_pvalue(panel, 0.2, n_perm=1000, seed=570) < 0.05


def test_impairment_link_present_and_survives_null():
    # High goodwill impairs more, at the effect AND at the return-null (imp_beta drives it).
    for alpha in (-0.14, 0.0):
        panel, _ = data.synthetic_panel(ret_alpha=alpha, seed=570)
        gap = st.impairment_gap(st.bucket_returns(panel, 0.2))
        assert gap["gap"] > 0  # high bucket impairs more


def test_flat_at_the_null():
    panel, truth = data.synthetic_panel(ret_alpha=0.0, seed=570)
    assert not truth.has_effect
    b = st.bucket_returns(panel, 0.2)
    ls = st.long_short_tstat(b)
    reg = st.cross_section_regression(panel)
    assert abs(ls["t"]) < 2.0
    assert abs(reg["slope_t"]) < 2.0


def test_seed_robust_control():
    null = st.synthetic_mean_t(data, ret_alpha=0.0, n_seeds=20, base_seed=570)
    strong = st.synthetic_mean_t(data, ret_alpha=-0.14, n_seeds=20, base_seed=570)
    # flat at null, clearly negative firm slope-t when planted
    assert abs(null["slope_t"]) < 1.0
    assert strong["slope_t"] < -3.0
    assert strong["ls_t"] > 3.0


def test_costs_reduce_spread():
    panel, _ = data.synthetic_panel(ret_alpha=-0.14, seed=570)
    b = st.bucket_returns(panel, 0.2)
    gross = b["spread"]
    net = st.net_spread(b, cost_bps=5.0, borrow_ann_bps=150.0, holding_years=1.0)
    assert net < gross


def test_real_fetch_raises_by_design():
    assert data.fetch_panel(fetch=False).empty
    try:
        data.fetch_panel(fetch=True)
    except NotImplementedError:
        return
    raise AssertionError("fetch_panel(fetch=True) must raise — the study is synthetic-only")
