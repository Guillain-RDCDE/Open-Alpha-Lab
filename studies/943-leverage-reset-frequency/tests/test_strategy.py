"""Replication invariants, inference primitives, and the study's spine — all offline.

The spine: the monthly-minus-daily reset gap is *positive* in a planted choppy world,
*negative* in a planted trending world, and centred at zero on the iid null. A second,
deliberately awkward test pins the honest caveat — the contemporaneous efficiency-ratio
slope is strongly negative **even on the null**, so that decomposition is arithmetic
about path shape, not evidence of a signal.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reset_freq import data, strategy as st  # noqa: E402


def _frame(prices, spread_bps=0.0):
    return st.synthetic_frame(prices, spread_bps=spread_bps)


def _gap(prices, leverage=2.0):
    """Mean daily monthly-minus-daily reset gap, in bps (liquidation disabled)."""
    fr = _frame(prices)
    d = st.replicate(fr["asset"], fr["fin"], leverage, "daily", maintenance=0.0)
    m = st.replicate(fr["asset"], fr["fin"], leverage, "monthly", maintenance=0.0)
    return float((m["r"] - d["r"]).mean() * 1e4)


# --------------------------------------------------------------------------- #
# Financing
# --------------------------------------------------------------------------- #
def test_financing_rate_uses_the_previous_close():
    irx = pd.Series([5.0, 5.0, 1.0], index=pd.bdate_range("2020-01-01", periods=3))
    f = st.financing_rate(irx, spread_bps=0.0)
    assert np.isnan(f.iloc[0])                       # nothing known before the first close
    assert abs(f.iloc[1] - 0.05 / 252) < 1e-12       # yesterday's 5%, not today's
    assert abs(f.iloc[2] - 0.05 / 252) < 1e-12


def test_financing_spread_adds_linearly():
    irx = pd.Series([2.0, 2.0, 2.0], index=pd.bdate_range("2020-01-01", periods=3))
    a = st.financing_rate(irx, spread_bps=0.0).iloc[-1]
    b = st.financing_rate(irx, spread_bps=100.0).iloc[-1]
    assert abs((b - a) - 0.01 / 252) < 1e-12


# --------------------------------------------------------------------------- #
# Replication invariants
# --------------------------------------------------------------------------- #
def test_daily_reset_pins_exposure(choppy):
    fr = _frame(choppy[0])
    d = st.replicate(fr["asset"], fr["fin"], 3.0, "daily", maintenance=0.0)
    assert np.allclose(d["w"].to_numpy(), 3.0)
    assert bool(d["alive"].all())


def test_daily_reset_matches_the_closed_form_when_costless(choppy):
    fr = _frame(choppy[0])
    d = st.replicate(fr["asset"], fr["fin"], 2.0, "daily", cost_bps=0.0, maintenance=0.0)
    expected = 2.0 * fr["asset"] - 1.0 * fr["fin"]
    assert np.allclose(d["r"].to_numpy(), expected.to_numpy())


def test_monthly_reset_returns_to_target_each_month(choppy):
    fr = _frame(choppy[0])
    m = st.replicate(fr["asset"], fr["fin"], 2.0, "monthly", maintenance=0.0)
    month = fr.index.to_period("M")
    first_of_month = pd.Series(month, index=fr.index).ne(
        pd.Series(month, index=fr.index).shift(1))
    w_first = m["w"][first_of_month.to_numpy()]
    assert np.allclose(w_first.to_numpy(), 2.0)
    # ...and it genuinely drifts in between (otherwise the two arms would be identical).
    assert m["w"].std(ddof=1) > 0.01


def test_monthly_reset_exposure_drifts_the_right_way():
    """A falling asset must raise effective leverage between resets, and vice versa."""
    idx = pd.bdate_range("2020-01-02", periods=40)
    down = pd.Series(-0.01, index=idx)
    up = pd.Series(+0.01, index=idx)
    fin = pd.Series(0.0, index=idx)
    w_down = st.replicate(down, fin, 2.0, "monthly", cost_bps=0.0, maintenance=0.0)["w"]
    w_up = st.replicate(up, fin, 2.0, "monthly", cost_bps=0.0, maintenance=0.0)["w"]
    assert w_down.iloc[15] > 2.0
    assert w_up.iloc[15] < 2.0


def test_execution_lag_reset_applies_the_day_after_month_end():
    """The reset decided at the month-end close is in force from the NEXT session."""
    idx = pd.bdate_range("2020-01-27", periods=10)      # crosses into February
    r = pd.Series(-0.02, index=idx)
    fin = pd.Series(0.0, index=idx)
    path = st.replicate(r, fin, 2.0, "monthly", cost_bps=0.0, maintenance=0.0)
    month = idx.to_period("M")
    last_jan = int(np.max(np.where(month == month[0])[0]))
    assert path["w"].iloc[last_jan] > 2.0              # still drifted on the reset day
    assert abs(path["w"].iloc[last_jan + 1] - 2.0) < 1e-12   # back on target the next day


def test_no_lookahead_future_moves_do_not_change_past_returns(choppy):
    fr = _frame(choppy[0])
    base = st.replicate(fr["asset"], fr["fin"], 2.0, "monthly", maintenance=0.0)
    perturbed_asset = fr["asset"].copy()
    perturbed_asset.iloc[1500:] += 0.05
    pert = st.replicate(perturbed_asset, fr["fin"], 2.0, "monthly", maintenance=0.0)
    assert np.allclose(base["r"].to_numpy()[:1500], pert["r"].to_numpy()[:1500])


def test_costs_monotonically_reduce_return(choppy):
    fr = _frame(choppy[0])
    means = [st.replicate(fr["asset"], fr["fin"], 2.0, "daily", cost_bps=c,
                          maintenance=0.0)["r"].mean() for c in (0.0, 2.0, 10.0)]
    assert means[0] >= means[1] >= means[2]


def test_financing_cost_reduces_return(choppy):
    fr_cheap = _frame(choppy[0], spread_bps=0.0)
    fr_dear = _frame(choppy[0], spread_bps=300.0)
    a = st.replicate(fr_cheap["asset"], fr_cheap["fin"], 2.0, "daily", maintenance=0.0)["r"].mean()
    b = st.replicate(fr_dear["asset"], fr_dear["fin"], 2.0, "daily", maintenance=0.0)["r"].mean()
    assert a > b


# --------------------------------------------------------------------------- #
# Margin calls and wipeouts — the hazard only the monthly reset carries
# --------------------------------------------------------------------------- #
def test_maintenance_margin_liquidates_a_drifting_monthly_account():
    idx = pd.bdate_range("2020-01-02", periods=20)
    crash = pd.Series(-0.02, index=idx)
    fin = pd.Series(0.0, index=idx)
    called = st.replicate(crash, fin, 3.0, "monthly", cost_bps=0.0, maintenance=0.25)
    patient = st.replicate(crash, fin, 3.0, "monthly", cost_bps=0.0, maintenance=0.0)
    assert called.attrs["liquidation_date"] is not None
    assert not bool(called["alive"].iloc[-1])
    assert float(called["r"].iloc[-1]) == 0.0          # no cash leg passed => 0% by default
    assert patient.attrs["liquidation_date"] is None   # an infinitely patient lender
    assert patient["w"].max() > called["w"].max()      # ...who lets exposure run further


def test_a_called_account_earns_the_cash_leg_not_zero():
    """After the call the proceeds sit in cash — so the arm's EXCESS return is 0, not −cash.

    Crediting 0% to a liquidated arm would charge it the cash rate every day for the rest
    of the sample, an invented drag on top of the equity the call actually destroyed. In
    an excess-of-cash race that is a thumb on the scale, so ``replicate`` takes the cash
    leg and the excess series is exactly flat after the call.
    """
    idx = pd.bdate_range("2020-01-02", periods=30)
    crash = pd.Series(-0.02, index=idx)
    fin = pd.Series(0.0, index=idx)
    cash = pd.Series(0.0002, index=idx)                # ~5%/yr cash leg
    p = st.replicate(crash, fin, 3.0, "monthly", cost_bps=0.0, maintenance=0.25, cash_r=cash)
    liq = int(np.where(~p["alive"].to_numpy())[0][0])  # first dead day
    assert np.allclose(p["r"].to_numpy()[liq:], 0.0002)
    excess = (p["r"] - cash).to_numpy()
    assert np.allclose(excess[liq:], 0.0)
    # A wiped-out account has nothing left to deposit, so it earns nothing either way.
    wipe = st.replicate(pd.Series([-0.60] + [0.0] * 9, index=pd.bdate_range("2020-01-02", periods=10)),
                        pd.Series(0.0, index=pd.bdate_range("2020-01-02", periods=10)),
                        3.0, "monthly", cost_bps=0.0, maintenance=0.0,
                        cash_r=pd.Series(0.0002, index=pd.bdate_range("2020-01-02", periods=10)))
    assert wipe.attrs["wiped_out"] is True
    assert np.allclose(wipe["r"].to_numpy()[1:], 0.0)


def test_daily_reset_never_gets_called_at_moderate_leverage():
    idx = pd.bdate_range("2020-01-02", periods=60)
    crash = pd.Series(-0.03, index=idx)
    fin = pd.Series(0.0, index=idx)
    d = st.replicate(crash, fin, 2.0, "daily", cost_bps=0.0, maintenance=0.25)
    assert d.attrs["liquidation_date"] is None
    assert bool(d["alive"].all())


def test_wipeout_is_floored_at_minus_one_hundred_percent():
    idx = pd.bdate_range("2020-01-02", periods=5)
    crash = pd.Series([-0.60, -0.10, -0.10, -0.10, -0.10], index=idx)
    fin = pd.Series(0.0, index=idx)
    path = st.replicate(crash, fin, 3.0, "monthly", cost_bps=0.0, maintenance=0.0)
    assert float(path["r"].iloc[0]) == -1.0
    assert not bool(path["alive"].iloc[1])
    assert float((1.0 + path["r"]).cumprod().iloc[-1]) == 0.0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_ols_hac_recovers_a_planted_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 800)
    y = 0.5 + 2.0 * x + rng.normal(0, 1, 800)
    fit = st.ols_hac(y, x)
    assert abs(fit["slope"] - 2.0) < 0.15
    assert fit["t"] > 10


def test_ols_hac_is_quiet_on_noise():
    rng = np.random.default_rng(3)
    fit = st.ols_hac(rng.normal(0, 1, 800), rng.normal(0, 1, 800))
    assert abs(fit["t"]) < 3


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_welch_and_one_sample_finite(choppy):
    fr = _frame(choppy[0])
    d = st.replicate(fr["asset"], fr["fin"], 2.0, "daily", maintenance=0.0)["r"]
    m = st.replicate(fr["asset"], fr["fin"], 2.0, "monthly", maintenance=0.0)["r"]
    assert np.isfinite(st.one_sample_t(m.to_numpy()))
    assert np.isfinite(st.welch_t(m.to_numpy(), d.to_numpy()))
    assert np.isfinite(st.sharpe_diff_tstat(m, d))


def test_summary_reports_the_expected_keys(choppy):
    fr = _frame(choppy[0])
    s = st.summary(st.replicate(fr["asset"], fr["fin"], 2.0, "daily", maintenance=0.0)["r"])
    for k in ("n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "terminal_wealth", "tstat"):
        assert k in s and np.isfinite(s[k])


def test_bootstrap_cis_bracket_their_point_estimates(choppy):
    fr = _frame(choppy[0])
    d = st.replicate(fr["asset"], fr["fin"], 2.0, "daily", maintenance=0.0)["r"] - fr["cash"]
    m = st.replicate(fr["asset"], fr["fin"], 2.0, "monthly", maintenance=0.0)["r"] - fr["cash"]
    ci = st.bootstrap_sharpe_ci(m, n_boot=400, seed=943)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]
    dci = st.bootstrap_diff_ci(m, d, n_boot=400, seed=943)
    assert dci["ci_low"] <= dci["adv"] <= dci["ci_high"]


# --------------------------------------------------------------------------- #
# Path-shape machinery
# --------------------------------------------------------------------------- #
def test_efficiency_ratio_bounds():
    straight = pd.Series([0.01] * 20)
    zigzag = pd.Series([0.01, -0.01] * 10)
    assert abs(st.efficiency_ratio(straight) - 1.0) < 1e-12
    assert st.efficiency_ratio(zigzag) < 1e-12


def test_monthly_gap_table_shape_and_bounds(choppy):
    tbl = st.monthly_gap_table(_frame(choppy[0]), 2.0, maintenance=0.0)
    assert {"gap_pp", "er", "mret_pp", "n"}.issubset(tbl.columns)
    assert len(tbl) > 100
    assert tbl["er"].between(0.0, 1.0).all()
    assert (tbl["n"] >= 15).all()


# --------------------------------------------------------------------------- #
# The study's spine — the planted effect and the null
# --------------------------------------------------------------------------- #
def test_choppy_world_favours_the_monthly_reset(choppy):
    assert _gap(choppy[0]) > 0.2


def test_trending_world_favours_the_daily_reset(trending):
    assert _gap(trending[0]) < -0.2


def test_null_world_has_no_reset_frequency_edge(null):
    assert abs(_gap(null[0])) < 0.2


def test_null_is_centred_across_seeds():
    gaps = np.array([
        _gap(data.synthetic_daily(phi=-0.15, signal_strength=0.0, n_years=12, seed=943 + s)[0])
        for s in range(6)
    ])
    assert abs(gaps.mean()) < 0.15
    assert (np.abs(gaps) >= 0.3).sum() <= 1


def test_the_gap_scales_with_leverage(choppy):
    """The reset-frequency gap is a leverage effect: bigger at 3x than at 2x."""
    assert _gap(choppy[0], leverage=3.0) > _gap(choppy[0], leverage=2.0) > 0


def test_contemporaneous_chop_slope_is_mechanical_even_on_the_null(null):
    """The honest caveat, pinned as a test.

    The efficiency-ratio slope is strongly negative on an *iid random walk* too — months
    that happened to be choppy always favour the monthly reset, by arithmetic. So the
    contemporaneous slope decomposes a path; it is never evidence of an edge. Only the
    unconditional mean gap can be (and on the null it is zero).
    """
    tbl = st.monthly_gap_table(_frame(null[0]), 2.0, maintenance=0.0)
    fit = st.chop_regression(tbl)
    assert fit["slope"] < -0.5 and fit["t"] < -3
    assert abs(_gap(null[0])) < 0.2


def test_synthetic_detect_signs_match_the_planted_worlds(choppy, trending, null):
    assert st.synthetic_detect(choppy[0])["mean_gap_bps"] > 0
    assert st.synthetic_detect(trending[0])["mean_gap_bps"] < 0
    assert abs(st.synthetic_detect(null[0])["mean_gap_bps"]) < 0.2


def test_race_and_era_cut_run_on_a_synthetic_frame(choppy):
    fr = _frame(choppy[0])
    r = st.race(fr, 2.0, maintenance=0.0)
    for k in ("sharpe_adv", "t_diff", "mean_diff_bps", "w_mean", "w_max"):
        assert np.isfinite(r[k])
    eras = st.era_cut(fr, 2.0, split="2012-01-01", maintenance=0.0)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["sharpe_adv"])


def test_predictive_regression_runs_and_reports_a_switch_rule(choppy):
    tbl = st.monthly_gap_table(_frame(choppy[0]), 2.0, maintenance=0.0)
    pf = st.predictive_regression(tbl)
    assert np.isfinite(pf["slope"]) and np.isfinite(pf["t"])
    assert pf["switch_n"] == len(tbl) - 1
