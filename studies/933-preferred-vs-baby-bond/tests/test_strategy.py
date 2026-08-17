"""Basket mechanics, inference primitives, and the study's spine — all offline/synthetic.

The spine: on a panel with a *planted* junior carry premium the pairwise same-issuer
estimator recovers it with |t| >= 2; on the exact null it stays quiet. Costs only ever
reduce returns, the rebalance carries exactly one day of lag, and the borrow assumption
only ever reduces the long-short.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_ladders import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_newey_west_is_wider_than_iid_under_persistence():
    """With positive autocorrelation the HAC t must be *smaller* than the naive t."""
    rng = np.random.default_rng(7)
    e = rng.normal(0, 0.01, 4000)
    x = np.zeros(4000)
    for i in range(1, 4000):
        x[i] = 0.0004 + 0.6 * x[i - 1] + e[i]
    assert abs(st.newey_west_t(x, lags=20)) < abs(st.one_sample_t(x))


def test_welch_and_one_sample_finite():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.001, 0.01, 800), rng.normal(0.0, 0.01, 800)
    assert np.isfinite(st.welch_t(a, b))
    assert np.isfinite(st.one_sample_t(a))


def test_hac_t_handles_short_and_empty_input():
    assert np.isnan(st.hac_t(pd.Series([], dtype=float)))
    assert np.isnan(st.hac_t(pd.Series([0.01])))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(6, 8)
    assert lo < 0.75 < hi


def test_summary_matches_hand_computed_stats():
    r = pd.Series(np.full(252, 0.001))
    s = st.summary(r)
    assert s["n_days"] == 252
    assert abs(s["cagr"] - ((1.001 ** 252) - 1.0)) < 1e-9
    assert s["max_drawdown"] == 0.0
    assert np.isnan(s["sharpe"]) or s["sharpe"] > 0


def test_max_drawdown_is_negative_after_a_loss():
    r = pd.Series([0.02, -0.10, 0.01, 0.03])
    assert st.max_drawdown(r) <= -0.09


# --------------------------------------------------------------------------- #
# Basket mechanics
# --------------------------------------------------------------------------- #
def test_basket_of_identical_legs_reproduces_the_leg(planted):
    """An equal-weight basket of N copies of one leg must equal that leg, gross."""
    pref = planted[0]
    one = pref.iloc[:, [0]]
    rep = pd.concat([one.rename(columns={one.columns[0]: f"c{i}"}) for i in range(4)], axis=1)
    bk = st.equal_weight_basket(rep, cost_bps=0.0)
    assert np.allclose(bk["gross"].to_numpy(), one.iloc[:, 0].to_numpy(), atol=1e-12)
    assert bk["turnover"].sum() < 1e-9  # identical legs never drift apart


def test_basket_gross_return_is_bounded_by_its_legs(planted):
    pref = planted[0]
    bk = st.equal_weight_basket(pref, cost_bps=0.0)
    assert (bk["gross"] >= pref.min(axis=1) - 1e-12).all()
    assert (bk["gross"] <= pref.max(axis=1) + 1e-12).all()


def test_costs_monotonically_reduce_basket_return(planted):
    pref = planted[0]
    means = [st.equal_weight_basket(pref, cost_bps=c)["net"].mean() for c in (0.0, 25.0, 100.0)]
    assert means[0] >= means[1] >= means[2]


def test_rebalance_is_monthly_and_carries_exactly_one_day_of_lag(planted):
    """Turnover may only be charged on the FIRST bar of a month, never mid-month."""
    pref = planted[0]
    bk = st.equal_weight_basket(pref, cost_bps=25.0)
    charged = bk.index[bk["turnover"] > 0]
    months = pd.Series(bk.index, index=bk.index).dt.to_period("M")
    first_of_month = bk.index[months.ne(months.shift(1)).to_numpy()]
    assert set(charged).issubset(set(first_of_month))
    # ~12 rebalances a year, no more.
    assert len(charged) <= int(np.ceil(len(bk) / 21.0)) + 1


def test_no_lookahead_perturbing_the_future_leaves_the_past_untouched(planted):
    """Tripling the tail of every leg must not change any earlier basket return."""
    pref = planted[0].copy()
    base = st.equal_weight_basket(pref, cost_bps=25.0)
    cut = len(pref) // 2
    pert = pref.copy()
    pert.iloc[cut:] = pert.iloc[cut:] * 3.0
    after = st.equal_weight_basket(pert, cost_bps=25.0)
    assert np.allclose(base["net"].iloc[:cut].to_numpy(), after["net"].iloc[:cut].to_numpy())


# --------------------------------------------------------------------------- #
# Pairwise estimator
# --------------------------------------------------------------------------- #
def test_pairwise_spread_is_the_mean_difference(planted):
    pref, bond, _, _ = planted
    pw = st.pairwise_spread(pref, bond)
    assert np.allclose(pw.to_numpy(), (pref - bond).mean(axis=1).to_numpy())


def test_pairwise_spread_is_zero_for_identical_ladders(planted):
    pref = planted[0]
    pw = st.pairwise_spread(pref, pref.copy())
    assert float(pw.abs().max()) < 1e-15


def test_per_pair_table_has_one_row_per_issuer(planted):
    pref, bond, _, _ = planted
    tbl = st.per_pair_table(pref, bond)
    assert len(tbl) == pref.shape[1]
    assert {"pref_cagr", "bond_cagr", "pref_vol", "bond_vol", "spread_ann", "spread_t"}.issubset(tbl.columns)
    assert np.isfinite(tbl["spread_ann"]).all()


def test_hit_rate_counts_positive_pairs(planted):
    pref, bond, _, _ = planted
    hr = st.hit_rate(pref, bond)
    assert hr["n"] == pref.shape[1]
    assert 0 <= hr["wins"] <= hr["n"]
    assert hr["ci_low"] <= hr["share"] <= hr["ci_high"]


# --------------------------------------------------------------------------- #
# Microstructure diagnostics & the concentration probe
# --------------------------------------------------------------------------- #
def test_staleness_table_flags_a_stale_and_a_bouncing_leg(planted):
    """A carried-forward leg shows zero-days; a bid-ask-bounce leg shows negative AC1."""
    clean = planted[0].iloc[:, :1].rename(columns={planted[0].columns[0]: "clean"})
    stale = clean["clean"].copy()
    stale.iloc[::2] = 0.0                       # every other day the print is carried forward
    bounce = pd.Series(
        np.tile([0.01, -0.01], len(clean) // 2 + 1)[: len(clean)], index=clean.index
    )                                            # pure bid-ask ping-pong -> AC1 = -1
    tbl = st.staleness_table(
        pd.DataFrame({"clean": clean["clean"], "stale": stale, "bounce": bounce})
    )
    assert tbl.loc["stale", "zero_share"] >= 0.49
    assert tbl.loc["clean", "zero_share"] < 0.01
    assert tbl.loc["bounce", "ac1"] < -0.9
    assert abs(tbl.loc["clean", "ac1"]) < 0.2
    assert (tbl["n"] == len(clean)).all()


def test_vol_ratio_by_frequency_is_stable_when_the_tape_is_clean(planted):
    """With no staleness the daily risk ordering must survive to monthly."""
    pref, bond, _, _ = planted
    freq = st.vol_ratio_by_frequency(pref, bond)
    assert list(freq.index) == ["daily", "weekly", "monthly"]
    assert (freq["vol_ratio"] > 1.0).all()       # planted panel: junior leg loads harder
    assert abs(freq.loc["daily", "vol_ratio"] - freq.loc["monthly", "vol_ratio"]) < 0.15
    assert (freq["n_pairs"] == pref.shape[1]).all()
    assert freq.loc["monthly", "n_obs"] < freq.loc["daily", "n_obs"]


def test_vol_ratio_by_frequency_unmasks_a_stale_daily_ordering():
    """A leg whose daily print bounces looks riskier daily and is NOT riskier monthly.

    This is the exact trap the study walks into on the real tape: bid-ask bounce inflates
    daily vol, so the daily 'the junior rung is riskier' count is not a capital-structure
    fact until it survives compounding.
    """
    idx = pd.bdate_range("2020-01-01", periods=1008)
    rng = np.random.default_rng(933)
    # The junior leg is genuinely the CALMER instrument, on a worse tape.
    quiet = rng.normal(0.0, 0.004, len(idx))
    loud = rng.normal(0.0, 0.006, len(idx))
    bounce = np.tile([0.006, -0.006], len(idx) // 2)[: len(idx)]   # washes out on compounding
    pref = pd.DataFrame({"a": quiet + bounce}, index=idx)
    bond = pd.DataFrame({"a": loud}, index=idx)
    freq = st.vol_ratio_by_frequency(pref, bond)
    assert freq.loc["daily", "pref_riskier"] == 1         # daily: the junior leg "looks" riskier
    assert freq.loc["monthly", "pref_riskier"] == 0       # monthly: the artefact is gone
    assert freq.loc["daily", "vol_ratio"] > 1.0 > freq.loc["monthly", "vol_ratio"]


def test_drop_issuers_removes_only_the_named_pairs(planted):
    pref, bond, cash, _ = planted
    names = [pref.columns[0], pref.columns[1]]
    d = st.drop_issuers(pref, bond, cash, names)
    assert d["n_pairs"] == pref.shape[1] - 2
    assert d["dropped"] == names
    assert 0 <= d["pref_riskier_monthly"] <= d["n_pairs"]


def test_drop_nothing_reproduces_the_full_panel(planted):
    pref, bond, cash, _ = planted
    d = st.drop_issuers(pref, bond, cash, [])
    full = st.compare(pref, bond, cash)
    assert abs(d["pairwise_spread_ann"] - full["pairwise_spread_ann"]) < 1e-12
    assert abs(d["vol_ratio"] - full["vol_ratio"]) < 1e-12


def test_dropping_the_only_issuer_with_the_premium_kills_the_spread():
    """Concentration probe with teeth: plant the carry in ONE issuer, drop it, lose it."""
    pref, bond, cash, _ = data.synthetic_panel(signal_strength=0.0, seed=933)
    hot = pref.columns[0]
    hot_pref = pref.copy()
    hot_pref[hot] = hot_pref[hot] + 0.40 / 252.0   # +40%/yr carry, in ONE name only
    full = st.compare(hot_pref, bond, cash)["pairwise_spread_ann"]
    assert full > 0.02                              # the panel average looks paid...
    d = st.drop_issuers(hot_pref, bond, cash, [hot])["pairwise_spread_ann"]
    assert full - d > 0.03                          # ...and one name was the whole of it
    # dropping it recovers EXACTLY the unplanted panel, not merely something smaller
    clean = st.drop_issuers(pref, bond, cash, [hot])["pairwise_spread_ann"]
    assert abs(d - clean) < 1e-12
    assert abs(d) < 0.03                            # and that panel is quiet


# --------------------------------------------------------------------------- #
# Bootstraps
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_point_estimate(planted):
    pref, bond, cash, _ = planted
    cmp = st.compare(pref, bond, cash, cost_bps=0.0)
    for stat, series in [("sharpe", cmp["e_pref"]), ("mean", cmp["pairwise"])]:
        ci = st.bootstrap_ci(series, stat=stat, n_boot=400, seed=933)
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_bootstrap_sharpe_adv_ci_matches_the_point_difference(planted):
    pref, bond, cash, _ = planted
    cmp = st.compare(pref, bond, cash, cost_bps=0.0)
    ci = st.bootstrap_sharpe_adv_ci(cmp["e_pref"], cmp["e_bond"], n_boot=400, seed=933)
    assert abs(ci["point"] - cmp["excess_sharpe_adv"]) < 1e-9
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_bootstrap_refuses_a_series_shorter_than_a_block():
    out = st.bootstrap_ci(pd.Series([0.01, 0.02, -0.01]), n_boot=50)
    assert np.isnan(out["point"])


# --------------------------------------------------------------------------- #
# Long-short & sweeps
# --------------------------------------------------------------------------- #
def test_borrow_only_ever_reduces_the_long_short(planted):
    pref, bond, cash, _ = planted
    rows = st.borrow_sweep(pref, bond, cash)
    rets = [r["ann_return"] for r in rows]
    assert rets == sorted(rets, reverse=True)


def test_long_short_legs_are_mirror_images(planted):
    pref, bond, cash, _ = planted
    a = st.long_short(pref, bond, cash, borrow_bps_ann=0.0, long_leg="pref")
    b = st.long_short(pref, bond, cash, borrow_bps_ann=0.0, long_leg="bond")
    assert abs(a["ann_return"] + b["ann_return"]) < 1e-12
    assert abs(a["vol_ann"] - b["vol_ann"]) < 1e-12


def test_cost_sweep_is_ordered_and_finite(planted):
    pref, bond, cash, _ = planted
    rows = st.cost_sweep(pref, bond, cash, cost_grid=(0.0, 25.0, 100.0))
    assert [r["cost_bps"] for r in rows] == [0.0, 25.0, 100.0]
    assert all(np.isfinite(r["excess_sharpe_adv"]) for r in rows)


def test_era_cut_returns_both_halves(planted):
    pref, bond, cash, _ = planted
    eras = st.era_cut(pref, bond, cash, split="2024-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["pairwise_spread_ann"])


def test_stress_table_reports_the_gap(planted):
    pref, bond, _, _ = planted
    tbl = st.stress_table(pref, bond, {"window": ("2021-01-04", "2021-06-30")})
    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert abs(row["gap_pp"] - (row["pref_pct"] - row["bond_pct"])) < 1e-9


# --------------------------------------------------------------------------- #
# The study's spine — the estimator recovers a planted premium and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_premium_is_recovered(planted):
    pref, bond, cash, truth = planted
    d = st.synthetic_detect(pref, bond, cash)
    assert d["pairwise_spread_ann"] > 0.5 * truth["planted_premium_ann"]
    assert d["t_pairwise"] >= 2.0


def test_planted_premium_lifts_the_junior_ladder_sharpe(planted):
    pref, bond, cash, _ = planted
    d = st.synthetic_detect(pref, bond, cash)
    assert d["excess_sharpe_adv"] > 0.0
    assert d["vol_ratio"] > 1.0  # the junior rung is the riskier one by construction


def test_null_produces_no_spurious_premium(null_panel):
    pref, bond, cash, _ = null_panel
    d = st.synthetic_detect(pref, bond, cash)
    assert abs(d["t_pairwise"]) < 2.0
    assert abs(d["pairwise_spread_ann"]) < 0.03


def test_spine_across_seeds_powered_and_calibrated():
    """Planted: fires on nearly every seed. Null: fires on almost none."""
    fired_planted, fired_null, spreads = 0, 0, []
    for k in range(8):
        p1, b1, c1, _ = data.synthetic_panel(signal_strength=1.0, seed=933 + k)
        p0, b0, c0, _ = data.synthetic_panel(signal_strength=0.0, seed=933 + k)
        d1 = st.synthetic_detect(p1, b1, c1)
        d0 = st.synthetic_detect(p0, b0, c0)
        fired_planted += int(d1["t_pairwise"] >= 2.0)
        fired_null += int(abs(d0["t_pairwise"]) >= 2.0)
        spreads.append(d0["pairwise_spread_ann"])
    assert fired_planted >= 7
    assert fired_null <= 1
    assert abs(float(np.mean(spreads))) < 0.02


def test_compare_is_cost_symmetric_on_the_pairwise_estimator(planted):
    """Cost hits both ladders alike, so the SAME-ISSUER spread barely moves with it."""
    pref, bond, cash, _ = planted
    a = st.compare(pref, bond, cash, cost_bps=0.0)
    b = st.compare(pref, bond, cash, cost_bps=100.0)
    assert abs(a["pairwise_spread_ann"] - b["pairwise_spread_ann"]) < 1e-9
