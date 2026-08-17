"""Race mechanics, tax-model invariants, and the study's spine — all offline/synthetic.

The spine: on a pair with a *planted* wrapper drag the race recovers the drag with
|HAC t| >= 2; on the null (identical wrappers plus tracking noise) it stays quiet. The tax
layer is a model, so it is tested as one: zero rates must be a no-op, higher rates must
cost more, and the deferral advantage must move with the payout assumption — which is
exactly why the study reports a swept range rather than a single after-tax number.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper_tax import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_newey_west_handles_degenerate_input():
    assert np.isnan(st.newey_west_t([1.0]))
    assert np.isnan(st.one_sample_t([1.0]))
    assert np.isnan(st.welch_t([1.0], [2.0, 3.0]))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(3, 11)
    assert lo < 3 / 11 < hi


def test_blended_1256_rate_is_the_statutory_60_40():
    assert st.blended_1256_rate(0.40, 0.20) == pytest.approx(0.6 * 0.20 + 0.4 * 0.40)


# --------------------------------------------------------------------------- #
# Returns, tracking, race mechanics
# --------------------------------------------------------------------------- #
def test_daily_returns_drop_the_entry_bar(planted):
    """The single execution lag: the position is entered at the first common close, so
    the first counted return is the next session's."""
    prices, _ = planted
    r = st.daily_returns(prices, ["ric", "k1", "cash"])
    assert len(r) == len(prices) - 1
    assert r.index[0] == prices.index[1]
    assert not r.isna().any().any()


def test_tracking_stats_on_identical_series_is_degenerate():
    idx = pd.bdate_range("2015-01-01", periods=500)
    a = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500), index=idx)
    trk = st.tracking_stats(a, a)
    assert trk["te_ann"] == pytest.approx(0.0)
    assert trk["corr"] == pytest.approx(1.0)
    assert trk["beta"] == pytest.approx(1.0)


def test_tracking_stats_recovers_a_constant_drag():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    base = pd.Series(np.random.default_rng(2).normal(0.0002, 0.01, 1000), index=idx)
    drag = 0.02 / 252
    trk = st.tracking_stats(base - drag, base)
    assert trk["diff_ann"] == pytest.approx(-0.02, abs=1e-6)
    assert trk["te_ann"] == pytest.approx(0.0, abs=1e-9)


def test_race_reports_both_legs_excess_of_cash(planted):
    prices, _ = planted
    res = st.race(prices, "ric", "k1", "cash")
    r = st.daily_returns(prices, ["ric", "k1", "cash"])
    assert res["e_a"].equals((r["ric"] - r["cash"]).rename("e_a"))
    assert res["e_b"].equals((r["k1"] - r["cash"]).rename("e_b"))
    assert res["n_days"] == len(r)


def test_extra_cost_only_hurts_the_charged_leg(planted):
    prices, _ = planted
    base = st.race(prices, "ric", "k1", "cash")
    costed = st.race(prices, "ric", "k1", "cash", extra_cost_bps_a=25.0)
    assert costed["diff_ann"] < base["diff_ann"]
    assert costed["b"]["sharpe"] == pytest.approx(base["b"]["sharpe"])


def test_cost_sweep_is_monotone(planted):
    prices, _ = planted
    rows = st.cost_sweep(prices, "ric", "k1", "cash")
    diffs = [r["diff_ann"] for r in rows]
    assert all(x >= y for x, y in zip(diffs, diffs[1:]))


def test_bootstrap_diff_ci_brackets_the_point(planted):
    prices, _ = planted
    res = st.race(prices, "ric", "k1", "cash")
    ci = st.bootstrap_diff_ci(res["r_a"], res["r_b"], n_boot=400, seed=915)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["boot_sd"] > 0


def test_bootstrap_sharpe_adv_ci_brackets_the_point(planted):
    prices, _ = planted
    res = st.race(prices, "ric", "k1", "cash")
    ci = st.bootstrap_sharpe_adv_ci(res["e_a"], res["e_b"], n_boot=400, seed=915)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_power_stats_reports_a_minimum_detectable_difference(planted):
    prices, _ = planted
    res = st.race(prices, "ric", "k1", "cash")
    pw = st.power_stats(res["r_a"], res["r_b"], boot_sd_ann=0.004)
    assert pw["mde_iid_ann"] == pytest.approx(2.0 * pw["se_iid_ann"])
    assert pw["mde_boot_ann"] == pytest.approx(0.008)
    assert pw["se_iid_ann"] > 0


def test_horizon_tracking_matches_daily_when_resampled_to_nothing(planted):
    prices, _ = planted
    mon = st.horizon_tracking(prices, "ric", "k1", freq="ME")
    assert mon["n_obs"] > 20
    assert mon["te_ann"] > 0
    # The synthetic difference is i.i.d., so daily-scaled and monthly TE broadly agree —
    # unlike the real tape, where the daily bounce inflates the daily-scaled figure.
    assert 0.5 < mon["te_ann"] / mon["te_ann_daily_scaled"] < 2.0


def test_era_cut_returns_both_halves(planted):
    prices, _ = planted
    eras = st.era_cut(prices, "ric", "k1", "cash", split="2020-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["diff_ann"]) and np.isfinite(e["t_diff"])


def test_calendar_year_table_drops_partial_years(planted):
    prices, _ = planted
    full = st.calendar_year_table(prices, "ric", "k1", "cash", complete_years_only=False)
    trimmed = st.calendar_year_table(prices, "ric", "k1", "cash")
    assert {"a_ret", "b_ret", "cash_ret", "diff_pp"}.issubset(trimmed.columns)
    assert len(trimmed) < len(full)
    assert len(trimmed) >= 8


def test_sign_test_counts_and_brackets(planted):
    prices, _ = planted
    tbl = st.calendar_year_table(prices, "ric", "k1", "cash")
    sgn = st.sign_test(tbl["diff_pp"])
    assert 0 <= sgn["wins"] <= sgn["n"] == len(tbl)
    assert sgn["ci_low"] <= sgn["win_rate"] <= sgn["ci_high"]


def test_dispersion_table_shape(planted):
    prices, _ = planted
    disp = st.dispersion_table(prices, ["ric", "k1"], "cash")
    assert {"cagr", "vol_ann", "excess_sharpe", "max_drawdown"}.issubset(disp.columns)
    assert len(disp) == 2


# --------------------------------------------------------------------------- #
# The after-tax MODEL — tested as a model, never as a measurement
# --------------------------------------------------------------------------- #
def _toy_years():
    idx = pd.Index([2015, 2016, 2017, 2018, 2019], name="year")
    return pd.DataFrame(
        {"a_ret": [0.10, -0.05, 0.08, 0.12, -0.02],
         "b_ret": [0.10, -0.05, 0.08, 0.12, -0.02],
         "cash_ret": [0.01, 0.02, 0.03, 0.04, 0.05]},
        index=idx,
    )


def test_zero_tax_rates_are_a_no_op():
    tbl = _toy_years()
    pre = float((1.0 + tbl["b_ret"]).prod())
    assert st.after_tax_terminal_k1(tbl["b_ret"], tbl["cash_ret"], 0.0, 0.0) == pytest.approx(pre)
    assert st.after_tax_terminal_ric(tbl["a_ret"], tbl["cash_ret"], 0.0, 0.0) == pytest.approx(pre)


def test_higher_rates_always_cost_more():
    tbl = _toy_years()
    low = st.after_tax_terminal_k1(tbl["b_ret"], tbl["cash_ret"], 0.24, 0.15)
    high = st.after_tax_terminal_k1(tbl["b_ret"], tbl["cash_ret"], 0.408, 0.238)
    assert high < low < float((1.0 + tbl["b_ret"]).prod())


def test_ric_deferral_erodes_as_the_payout_share_rises():
    """More ordinary distributions = less deferral = less after-tax wealth."""
    tbl = _toy_years()
    terms = [st.after_tax_terminal_ric(tbl["a_ret"], tbl["cash_ret"], 0.408, 0.238, p)
             for p in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert all(x >= y for x, y in zip(terms, terms[1:]))


def test_k1_pays_nothing_at_liquidation():
    """Annual mark-to-market steps the basis up, so the K-1 leg owes nothing on sale —
    the terminal value must not depend on how long the position is then held flat."""
    tbl = _toy_years()
    flat = pd.concat([tbl, pd.DataFrame({"a_ret": [0.0], "b_ret": [0.0], "cash_ret": [0.0]},
                                        index=pd.Index([2020], name="year"))])
    a = st.after_tax_terminal_k1(tbl["b_ret"], tbl["cash_ret"], 0.408, 0.238)
    b = st.after_tax_terminal_k1(flat["b_ret"], flat["cash_ret"], 0.408, 0.238)
    assert a == pytest.approx(b)


def test_after_tax_race_and_same_tape_regime_gap_agree_on_one_tape():
    """With the two legs given identical returns, the after-tax gap IS the regime gap."""
    tbl = _toy_years()
    race_gap = st.after_tax_race(tbl, 0.358, 0.188, payout_share=1.0)["gap_pp"]
    regime_gap = st.same_tape_regime_gap(tbl, 0.358, 0.188, payout_share=1.0)["regime_gap_pp"]
    assert race_gap == pytest.approx(regime_gap)


def test_bracket_sweep_covers_the_declared_grid():
    tbl = _toy_years()
    grid = st.bracket_sweep(tbl)
    assert len(grid) == len(st.BRACKETS) * len(st.PAYOUT_GRID)
    assert {"bracket", "payout_share", "gap_pp", "blended_1256"}.issubset(grid.columns)
    assert np.isfinite(grid["gap_pp"]).all()


# --------------------------------------------------------------------------- #
# The study's spine — the estimator recovers a real drag and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_drag_is_recovered(planted):
    prices, truth = planted
    det = st.synthetic_detect(prices, n_boot=300)
    assert det["estimated_drag_ann"] > 0
    assert abs(det["estimated_drag_ann"] - truth["planted_drag_ann"]) < 0.02
    assert det["t_diff"] <= -2.0                 # a drag shows as a negative difference


def test_planted_drag_ci_excludes_zero(planted):
    prices, _ = planted
    det = st.synthetic_detect(prices, n_boot=600)
    assert det["ci_high"] < 0.0


def test_null_produces_no_reliable_difference(null):
    prices, _ = null
    det = st.synthetic_detect(prices, n_boot=300)
    assert abs(det["t_diff"]) < 2.0
    assert det["ci_low"] < 0.0 < det["ci_high"]


def test_null_panel_is_centred_and_quiet():
    panel = data.synthetic_panel(n_pairs=8, signal_strength=0.0, seed=915)
    diffs, ts = [], []
    for prices, _ in panel:
        res = st.race(prices, "ric", "k1", "cash")
        diffs.append(res["diff_ann"])
        ts.append(res["t_diff"])
    diffs, ts = np.array(diffs), np.array(ts)
    assert abs(diffs.mean()) < 0.015                  # centred on zero, not on a "fee"
    assert (np.abs(ts) >= 2.0).sum() <= 1             # no systematic false positives


def test_signal_strength_orders_the_detected_drag():
    ests = []
    for s in (0.0, 0.5, 1.0):
        prices, _ = data.synthetic_daily(signal_strength=s, seed=915)
        ests.append(st.synthetic_detect(prices, n_boot=200)["estimated_drag_ann"])
    assert ests[0] < ests[1] < ests[2]
