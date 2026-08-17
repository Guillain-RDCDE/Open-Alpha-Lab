"""Basket mechanics, inference primitives, and the study's spine — all offline/synthetic.

The spine: on a panel where the fund skims a real fee, the do-it-yourself basket of top
holdings recovers roughly that fee with an HAC *t* far above 2; on the zero-fee null it
recovers nothing. In between sit the mechanical invariants that make that claim
trustworthy — one execution lag, monotone costs, renormalisation over live names, and a
tracking error that grows as the basket gets shallower.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diy_sector import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Rebalance calendar & the single execution lag
# --------------------------------------------------------------------------- #
def test_rebalance_flags_fire_on_the_first_day_of_each_period():
    idx = pd.bdate_range("2020-01-01", periods=260)
    flags = st.rebalance_flags(idx, "M")
    fired = idx[flags]
    # exactly one trade per month change, and never on the very first bar
    assert not flags[0]
    assert len(fired) == len(set(zip(fired.year, fired.month)))
    for d in fired:
        prev = idx[idx.get_loc(d) - 1]
        assert (prev.year, prev.month) != (d.year, d.month)


def test_rebalance_frequencies_are_ordered_by_turnover():
    idx = pd.bdate_range("2015-01-01", periods=1300)
    counts = {f: int(st.rebalance_flags(idx, f).sum()) for f in ("M", "Q", "A", "N")}
    assert counts["M"] > counts["Q"] > counts["A"] > counts["N"] == 0


def test_unknown_frequency_raises():
    with pytest.raises(ValueError):
        st.rebalance_flags(pd.bdate_range("2020-01-01", periods=10), "W")


def test_no_lookahead_perturbing_the_future_leaves_the_past_alone(small_panel):
    """Shifting the tail of the tape must not change any earlier basket return."""
    prices, truth = small_panel
    names = list(truth["weights"].index)
    base = st.basket_returns(prices[names], truth["weights"], cost_bps=5.0)
    pert = prices.copy()
    pert.loc[pert.index[600:], names] *= 3.0
    after = st.basket_returns(pert[names], truth["weights"], cost_bps=5.0)
    k = 590
    assert np.allclose(base["r_basket"].iloc[:k].to_numpy(),
                       after["r_basket"].iloc[:k].to_numpy())


# --------------------------------------------------------------------------- #
# Basket mechanics
# --------------------------------------------------------------------------- #
def test_basket_columns_and_no_nans(small_panel):
    prices, truth = small_panel
    bt = st.basket_returns(prices, truth["weights"], cost_bps=5.0)
    assert {"r_basket", "r_basket_gross", "turnover", "n_live"}.issubset(bt.columns)
    assert not bt[["r_basket", "r_basket_gross"]].isna().any().any()
    assert (bt["n_live"] == len(truth["weights"])).all()


def test_single_name_basket_reproduces_that_name(small_panel):
    """A one-name basket at zero cost must equal that name's own daily return."""
    prices, _ = small_panel
    w = pd.Series({"name_00": 1.0})
    bt = st.basket_returns(prices, w, cost_bps=0.0)
    ref = prices["name_00"].pct_change().reindex(bt.index)
    assert np.allclose(bt["r_basket"].to_numpy(), ref.to_numpy(), atol=1e-12)


def test_costs_monotonically_reduce_return(small_panel):
    prices, truth = small_panel
    means = [st.basket_returns(prices, truth["weights"], cost_bps=c)["r_basket"].mean()
             for c in (0.0, 5.0, 25.0, 100.0)]
    assert means[0] >= means[1] >= means[2] >= means[3]


def test_gross_leg_ignores_costs(small_panel):
    prices, truth = small_panel
    a = st.basket_returns(prices, truth["weights"], cost_bps=0.0)["r_basket_gross"]
    b = st.basket_returns(prices, truth["weights"], cost_bps=50.0)["r_basket_gross"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_turnover_falls_with_rebalance_frequency(small_panel):
    prices, truth = small_panel
    turns = [st.basket_returns(prices, truth["weights"], cost_bps=5.0, freq=f)["turnover"].sum()
             for f in ("M", "Q", "A", "N")]
    assert turns[0] > turns[1] > turns[2] > turns[3]
    assert turns[3] == pytest.approx(1.0)  # only the initial build


def test_late_listing_name_is_renormalised_not_dropped(small_panel):
    """A name that lists mid-sample must not distort the basket while it is absent."""
    prices, truth = small_panel
    names = list(truth["weights"].index)
    late = prices[names].copy()
    late.iloc[:300, late.columns.get_loc(names[-1])] = np.nan
    bt = st.basket_returns(late, truth["weights"], cost_bps=0.0)
    assert (bt["n_live"].iloc[:250] == len(names) - 1).all()
    assert (bt["n_live"].iloc[-50:] == len(names)).all()
    assert np.isfinite(bt["r_basket"]).all()


def test_basket_without_any_known_name_raises(small_panel):
    prices, _ = small_panel
    with pytest.raises(ValueError):
        st.basket_returns(prices, pd.Series({"not_a_name": 1.0}))


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_t_matches_one_sample_t_on_iid_data():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.normal(0.0005, 0.01, 4000))
    assert abs(st.hac_t(x) - st.one_sample_t(x.to_numpy())) < 0.5


def test_worst_rolling_shortfall_is_negative_and_bounded():
    rng = np.random.default_rng(3)
    d = pd.Series(rng.normal(0.0, 0.005, 1500))
    w = st.worst_rolling_shortfall(d)
    assert -1.0 < w < 0.0


def test_worst_rolling_shortfall_needs_a_full_window():
    assert np.isnan(st.worst_rolling_shortfall(pd.Series(np.zeros(50))))


def test_summary_on_a_constant_series():
    r = pd.Series(np.full(500, 0.0004))
    s = st.summary(r)
    assert s["max_drawdown"] == pytest.approx(0.0)
    assert s["cagr"] > 0
    assert np.isinf(s["sharpe"]) or s["sharpe"] > 100


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def test_race_reports_a_finite_picture(small_panel):
    prices, truth = small_panel
    res = st.race(prices, "fund", truth["weights"], prices["cash"], cost_bps=5.0)
    for k in ("ann_gap", "t_diff", "tracking_error", "corr", "beta",
              "sharpe_adv", "worst_12m_shortfall"):
        assert np.isfinite(res[k]), k
    assert 0.5 < res["corr"] <= 1.0


def test_tracking_error_grows_as_the_basket_gets_shallower(fee_panel):
    """Three names track a fifty-name fund worse than ten do — the whole trade-off."""
    prices, truth = fee_panel
    w10 = truth["weights"]
    w3 = (w10.iloc[:3] / w10.iloc[:3].sum())
    te10 = st.race(prices, "fund", w10, prices["cash"], cost_bps=0.0)["tracking_error"]
    te3 = st.race(prices, "fund", w3, prices["cash"], cost_bps=0.0)["tracking_error"]
    assert te3 > te10


def test_bootstrap_ci_brackets_the_point_estimate(small_panel):
    prices, truth = small_panel
    res = st.race(prices, "fund", truth["weights"], prices["cash"], cost_bps=5.0)
    ci = st.bootstrap_gap_ci(res["diff"], n_boot=400, seed=962)
    assert ci["ci_low"] <= ci["ann_gap"] <= ci["ci_high"]


def test_bootstrap_handles_a_series_shorter_than_a_block():
    out = st.bootstrap_gap_ci(pd.Series(np.zeros(5)))
    assert np.isnan(out["ann_gap"])


def test_era_cut_returns_both_halves(small_panel):
    prices, truth = small_panel
    split = str(prices.index[len(prices) // 2].date())
    eras = st.era_cut(prices, "fund", prices["cash"], truth["weights"], split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["ann_gap"]) and np.isfinite(e["tracking_error"])


def test_cost_sweep_is_monotone_in_the_gap(small_panel):
    prices, truth = small_panel
    rows = st.cost_sweep(prices, "fund", prices["cash"], truth["weights"],
                         cost_grid=(0.0, 10.0, 50.0))
    gaps = [r["ann_gap"] for r in rows]
    assert gaps[0] >= gaps[1] >= gaps[2]


def test_freq_sweep_covers_every_frequency(small_panel):
    prices, truth = small_panel
    rows = st.freq_sweep(prices, "fund", prices["cash"], truth["weights"])
    assert [r["freq"] for r in rows] == ["M", "Q", "A", "N"]
    assert all(np.isfinite(r["ann_gap"]) for r in rows)


def test_calendar_year_table_shape(small_panel):
    prices, truth = small_panel
    tbl = st.calendar_year_table(prices, "fund", prices["cash"], truth["weights"])
    assert {"diy_pct", "fund_pct", "gap_pp"}.issubset(tbl.columns)
    assert len(tbl) >= 4


def test_depth_sweep_runs_both_schemes_on_a_synthetic_stub(small_panel):
    """``depth_sweep`` is scheme-agnostic: it takes any weight function."""
    prices, truth = small_panel
    names = list(truth["weights"].index)

    def wfn(fund, n, scheme):
        sub = truth["weights"].iloc[:n] if scheme == "cap2026" else pd.Series(1.0, index=names[:n])
        return sub / sub.sum()

    rows = st.depth_sweep(prices, "fund", prices["cash"], wfn, depths=(3, 6))
    assert len(rows) == 4
    assert all(np.isfinite(r["tracking_error"]) for r in rows)


# --------------------------------------------------------------------------- #
# The study's spine — the detector is unbiased
# --------------------------------------------------------------------------- #
def test_planted_fee_is_recovered(fee_panel):
    prices, truth = fee_panel
    d = st.synthetic_detect(prices, truth, cost_bps=0.0)
    assert d["ann_gap"] > 0
    assert 0.7 < d["recovery_ratio"] < 1.3      # recovers the fee, not a multiple of it
    assert d["t_diff"] > 4.0


def test_null_panel_shows_no_gap(null_panel):
    prices, truth = null_panel
    d = st.synthetic_detect(prices, truth, cost_bps=0.0)
    assert abs(d["ann_gap"]) < 0.005
    assert abs(d["t_diff"]) < 2.0


def test_null_is_centred_across_seeds():
    gaps, ts = [], []
    for s in range(4):
        p, t = data.synthetic_panel(n_years=8, signal_strength=0.0, seed=962 + s)
        d = st.synthetic_detect(p, t, cost_bps=0.0)
        gaps.append(d["ann_gap"])
        ts.append(d["t_diff"])
    assert abs(float(np.mean(gaps))) < 0.005
    assert sum(abs(t) >= 2.0 for t in ts) <= 1


def test_costs_can_eat_a_small_planted_fee():
    """A fee smaller than the trading bill it takes to harvest it is not harvestable —
    the arithmetic behind the study's verdict."""
    p, t = data.synthetic_panel(n_years=8, fee_ann=0.0008, seed=962)
    gross = st.synthetic_detect(p, t, cost_bps=0.0)["ann_gap"]
    net = st.synthetic_detect(p, t, cost_bps=50.0)["ann_gap"]
    assert net < gross
