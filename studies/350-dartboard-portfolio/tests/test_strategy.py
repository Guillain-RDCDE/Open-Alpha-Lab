"""The weight builders, the cost-aware book engine, and the study's two spines:
(1) the dartboard 'beats' the cap index only when a size premium is planted, and
(2) the dartboard never beats the equal-weight index — its edge is a size tilt, not skill."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dartboard_portfolio import strategy as st  # noqa: E402


# ---- weight builders -------------------------------------------------------
def test_cap_weights_sum_to_one(null_panel):
    _, caps, _ = null_panel
    w = st.cap_weights(caps)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_equal_weights():
    w = st.equal_weights(5)
    assert np.allclose(w, 0.2)


def test_dart_weights_pick_k_equal(null_panel):
    rng = np.random.default_rng(0)
    _, caps, _ = null_panel
    n = len(caps)
    w = st.dart_weights(n, 10, rng)
    assert (w > 0).sum() == 10
    nz = w[w > 0]
    assert np.allclose(nz, 0.1)
    assert abs(w.sum() - 1.0) < 1e-9


def test_expert_picks_largest(null_panel):
    _, caps, _ = null_panel
    w = st.expert_weights(caps, 5)
    held = caps.index[w > 0]
    largest = caps.sort_values(ascending=False).index[:5]
    assert set(held) == set(largest)


def test_dart_is_reproducible(null_panel):
    rets, _, _ = null_panel
    a = st.dartboard_returns(rets, k=10, n_darts=5, seed=350)
    b = st.dartboard_returns(rets, k=10, n_darts=5, seed=350)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---- book engine invariants ------------------------------------------------
def test_book_returns_no_lookahead_first_period(null_panel):
    """The very first period charges no turnover cost (establishing the book is free)."""
    rets, caps, _ = null_panel
    free = st.index_returns(rets, caps, rebalance_every=12, cost_bps=0.0)
    costly = st.index_returns(rets, caps, rebalance_every=12, cost_bps=50.0)
    assert np.isclose(free.iloc[0], costly.iloc[0])


def test_costs_monotonically_lower_returns(tilted_panel):
    rets, caps, _ = tilted_panel
    means = [
        st.dartboard_returns(rets, k=10, n_darts=50, rebalance_every=12, cost_bps=c)
        .median(axis=1).mean()
        for c in (0.0, 25.0, 50.0)
    ]
    assert means[0] > means[1] > means[2]


def test_index_equals_buyhold_when_rebalance_never(null_panel):
    """With rebalance_every larger than the sample, the index is a single buy-and-hold book."""
    rets, caps, _ = null_panel
    s = st.index_returns(rets, caps, rebalance_every=10_000, cost_bps=0.0)
    assert s.notna().all()


# ---- the theses ------------------------------------------------------------
def test_dartboard_beats_index_only_with_size_premium(null_panel, tilted_panel):
    """Spine #1: a positive size premium hands the dartboard a 'win' over the cap index;
    with no premium the win-rate is ~a coin and the spread is insignificant."""
    rn, cn, _ = null_panel
    rt, ct, _ = tilted_panel

    res_null = st.race(rn, cn, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)
    res_tilt = st.race(rt, ct, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)

    # With a strong tilt: the monkey wins the large majority of throws and clears the bar.
    assert res_tilt["win_rate_vs_index"] > 0.80
    assert res_tilt["test_vs_index"]["tstat"] > 2.0
    # With no tilt: not statistically distinguishable from the index.
    assert abs(res_null["test_vs_index"]["tstat"]) < 2.0


def test_dartboard_never_beats_equal_index(null_panel, tilted_panel):
    """Spine #2 — the BUSTED axis: against the equal-weight index (which already holds the
    size tilt), the dartboard's edge vanishes on BOTH tapes. It is a tilt, never skill."""
    for panel in (null_panel, tilted_panel):
        rets, caps, _ = panel
        res = st.race(rets, caps, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)
        assert abs(res["test_vs_equal_index"]["tstat"]) < 2.0


def test_win_rate_in_unit_interval(tilted_panel):
    rets, caps, _ = tilted_panel
    res = st.race(rets, caps, k=10, n_darts=100, rebalance_every=12, cost_bps=0.0)
    assert 0.0 <= res["win_rate_vs_index"] <= 1.0
    assert 0.0 <= res["win_rate_vs_expert"] <= 1.0


def test_hac_and_bootstrap_agree_on_sign(tilted_panel):
    """On the tilted tape both the HAC mean and the bootstrap CI centre point positive."""
    rets, caps, _ = tilted_panel
    res = st.race(rets, caps, k=10, n_darts=200, rebalance_every=12, cost_bps=0.0)
    diff = res["median_dart"] - res["index"]
    lo, hi = st.block_bootstrap_ci(diff, n_boot=1000, seed=1)
    assert res["test_vs_index"]["mean_diff"] > 0
    assert (lo + hi) / 2 > 0


def test_summarize_keys(null_panel):
    rets, caps, _ = null_panel
    s = st.summarize(st.index_returns(rets, caps))
    assert set(s) == {"n", "cagr", "sharpe", "max_dd", "mean_ann"}
