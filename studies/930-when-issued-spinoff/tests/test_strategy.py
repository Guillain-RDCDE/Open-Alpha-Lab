"""Window logic, friction invariants, and the study's spine — all offline/synthetic.

The spine: on a planted world the estimator recovers both the parent run-up and the
child's regular-way drift with a clear t; on the null it recovers neither. Around that
sit the mechanical guarantees the audits care about — exactly one execution lag, no
parent window ever crossing the distribution, costs that only ever *reduce* a long
alpha, and a short leg that pays borrow.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from when_issued import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_recovers_a_mean():
    rng = np.random.default_rng(1)
    assert st.one_sample_t(0.01 + rng.normal(0, 0.02, 400)) > 3
    assert abs(st.one_sample_t(rng.normal(0, 0.02, 400))) < 3


def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(2)
    x = 0.001 + rng.normal(0, 0.01, 4000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 4000))) < 3


def test_newey_west_widens_se_under_positive_autocorrelation():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 0.01, 4000)
    x = 0.0008 + pd.Series(e).rolling(10).mean().dropna().to_numpy()
    assert abs(st.newey_west_t(x, lags=20)) < abs(st.one_sample_t(x))


def test_family_wise_p_never_flatters_the_single_test(planted):
    """Pricing five pre-specified legs can only make the best one look worse, never better."""
    prices, events, _ = planted
    panel = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    legs = ["a_parent_gross"] + [f"a_child_{h}_gross" for h in (5, 10, 21)] + ["a_comb_gross"]
    fw = st.family_wise_p(panel, legs, n_boot=800)
    assert fw["n_legs"] == 5
    assert 0.0 <= fw["p_single"] <= fw["p_fwe"] <= 1.0
    # the strongest leg it picks is the one with the largest |t|
    assert fw["best_leg"] == max(fw["t_obs"], key=lambda c: abs(fw["t_obs"][c]))


def test_family_wise_p_is_uniform_ish_on_the_null(null_world):
    prices, events, _ = null_world
    panel = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    legs = ["a_parent_gross"] + [f"a_child_{h}_gross" for h in (5, 10, 21)] + ["a_comb_gross"]
    fw = st.family_wise_p(panel, legs, n_boot=800)
    assert fw["p_fwe"] > 0.05  # a null family must not be condemned


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(8, 26)
    assert lo < 8 / 26 < hi


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(0.03, 0.08, 30)
    ci = st.bootstrap_mean_ci(x, n_boot=2000, seed=930)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0004, 0.01, 900)
    ci = st.block_bootstrap_mean_ci(x, n_boot=400, seed=930)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(6)
    x = rng.normal(0.0, 0.05, 40)
    a = st.bootstrap_mean_ci(x, n_boot=800, seed=7)
    b = st.bootstrap_mean_ci(x, n_boot=800, seed=7)
    c = st.bootstrap_mean_ci(x, n_boot=800, seed=8)
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_low"] != c["ci_low"]


# --------------------------------------------------------------------------- #
# Window construction — the one execution lag, and no window crossing the spin
# --------------------------------------------------------------------------- #
def test_parent_entry_is_strictly_after_the_announcement(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    assert len(panel) > 5
    assert (panel["entry"] > panel["announce"]).all()


def test_parent_exit_is_strictly_before_the_first_regular_way_session(planted):
    """No parent return may cross the distribution — that is where the tape is ambiguous."""
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    assert (panel["exit"] < panel["rw0"]).all()


def test_child_window_starts_on_or_after_the_stated_regular_way_date(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    assert (panel["rw0"] >= panel["regular_way"]).all()


def test_no_lookahead_perturbing_the_future_leaves_earlier_events_alone(planted):
    """Tripling every price after the last event must not move the earlier events' alphas."""
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    cut = panel["rw0"].iloc[-1]
    px2 = prices.copy()
    px2.loc[px2.index > cut] = px2.loc[px2.index > cut] * 3.0
    panel2 = st.event_panel(px2, events)
    a = panel["a_child_5_net"].iloc[:-1].to_numpy()
    b = panel2["a_child_5_net"].iloc[:-1].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_panel_is_sorted_by_distribution_date(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    assert panel["rw0"].is_monotonic_increasing


def test_alpha_is_asset_minus_benchmark(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    diff = panel["a_child_21_gross"] - (panel["r_child_21"] - panel["b_child_21"])
    assert diff.abs().max() < 1e-12


# --------------------------------------------------------------------------- #
# Friction invariants
# --------------------------------------------------------------------------- #
def test_costs_only_reduce_a_long_alpha(planted):
    prices, events, _ = planted
    means = []
    for c in (0.0, 10.0, 50.0):
        panel = st.event_panel(prices, events, cost_bps=c)
        means.append(panel["a_child_5_net"].mean())
    assert means[0] > means[1] > means[2]


def test_short_benchmark_leg_pays_borrow(planted):
    """A longer holding period must cost more borrow, all else equal."""
    prices, events, _ = planted
    free = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    paid = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=400.0)
    d5 = (free["a_child_5_net"] - paid["a_child_5_net"]).mean()
    d21 = (free["a_child_21_net"] - paid["a_child_21_net"]).mean()
    assert d5 > 0 and d21 > d5


def test_gross_and_net_differ_only_by_the_frictions(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events, cost_bps=10.0, bench_cost_bps=1.0, borrow_bps=40.0)
    expected = 2 * 10e-4 + 2 * 1e-4 + 40e-4 * 5 / st.TRADING_DAYS
    gap = (panel["a_child_5_gross"] - panel["a_child_5_net"]).to_numpy()
    assert np.allclose(gap, expected)


def test_short_side_economics_flip_the_sign_and_charge_borrow():
    prices, events, _ = data.synthetic_daily(signal_strength=1.0, seed=930)
    panel = st.event_panel(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    sh = st.short_child_economics(panel, horizon=21, cost_bps=0.0, bench_cost_bps=0.0,
                                  borrow_grid_pct=(0.0, 100.0))
    long_mean = panel["a_child_21_gross"].mean() * 100
    assert sh.loc[0.0, "mean_pct"] == pytest.approx(-long_mean, abs=1e-9)
    assert sh.loc[100.0, "mean_pct"] < sh.loc[0.0, "mean_pct"]


# --------------------------------------------------------------------------- #
# The combined leg and its ratio proxy
# --------------------------------------------------------------------------- #
def test_combined_weights_are_a_valid_split(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    w = panel["w_child"].dropna()
    assert ((w > 0) & (w < 1)).all()


def test_equal_weight_combined_uses_a_half(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events, equal_weight_combined=True)
    assert (panel["w_child"].dropna() == 0.5).all()


def test_ratio_multiplier_shifts_the_child_weight(planted):
    prices, events, _ = planted
    lo = st.event_panel(prices, events, ratio_mult=0.5)["w_child"].mean()
    hi = st.event_panel(prices, events, ratio_mult=2.0)["w_child"].mean()
    assert hi > lo


# --------------------------------------------------------------------------- #
# The daily hedged sleeve
# --------------------------------------------------------------------------- #
def test_sleeve_is_idle_outside_the_event_windows(planted):
    prices, events, _ = planted
    sl = st.hedged_sleeve(prices, events, window=21)
    assert 0.0 < sl["exposure"].mean() < 0.5
    idle = sl[sl["exposure"] == 0.0]
    assert (idle["r_long"].abs() < 1e-12).all()


def test_sleeve_active_days_scale_with_the_window(planted):
    prices, events, _ = planted
    a = st.hedged_sleeve(prices, events, window=5)["exposure"].sum()
    b = st.hedged_sleeve(prices, events, window=21)["exposure"].sum()
    assert b > a


def test_sleeve_costs_only_reduce_the_spread(planted):
    prices, events, _ = planted
    free = st.hedged_sleeve(prices, events, cost_bps=0.0, bench_cost_bps=0.0, borrow_bps=0.0)
    paid = st.hedged_sleeve(prices, events, cost_bps=25.0, bench_cost_bps=1.0, borrow_bps=100.0)
    assert paid["r_spread"].mean() < free["r_spread"].mean()


def test_sleeve_calendar_is_the_event_span_not_the_cache_span(planted):
    """The duty cycle must not depend on how much unrelated pre-history sits in the cache.

    Bolting five extra years of benchmark history onto the front of the frame must leave
    the sleeve's active fraction alone — otherwise 'idle 99% of the calendar' is a fact
    about the parquet files, not about the strategy.
    """
    prices, events, _ = planted
    base = st.hedged_sleeve(prices, events, window=5)
    pad = pd.DataFrame(
        {c: (100.0 if c in (data.BENCHMARK, data.CASH) else np.nan) for c in prices.columns},
        index=pd.bdate_range(end=prices.index[0] - pd.Timedelta(days=1), periods=1200),
    )
    padded = st.hedged_sleeve(pd.concat([pad, prices]), events, window=5)
    assert len(padded) == len(base)
    assert padded["exposure"].mean() == pytest.approx(base["exposure"].mean())


def test_sleeve_untrimmed_keeps_the_whole_calendar(planted):
    prices, events, _ = planted
    assert len(st.hedged_sleeve(prices, events, trim_to_events=False)) > \
        len(st.hedged_sleeve(prices, events, trim_to_events=True))


def test_sleeve_stats_are_finite(planted):
    prices, events, _ = planted
    s = st.sleeve_stats(st.hedged_sleeve(prices, events, window=5))
    for k in ("spread_mean_bps", "spread_t_hac", "spread_sharpe_active"):
        assert np.isfinite(s[k])


# --------------------------------------------------------------------------- #
# Robustness helpers
# --------------------------------------------------------------------------- #
def test_era_cut_splits_the_event_table(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events)
    mid = str(panel["rw0"].iloc[len(panel) // 2].date())
    eras = st.era_cut(panel, "a_child_21_net", split=mid)
    assert eras["early"]["n"] > 0 and eras["late"]["n"] > 0
    assert eras["early"]["n"] + eras["late"]["n"] == len(panel)


def test_sweeps_return_the_expected_shapes(planted):
    prices, events, _ = planted
    assert len(st.cost_sweep(prices, events, "a_child_5_net",
                             cost_grid=(0.0, 25.0), borrow_grid=(0.0, 40.0))) == 4
    assert len(st.announce_shift_sweep(prices, events, shifts=(-2, 0, 2))) == 3
    assert len(st.rw_shift_sweep(prices, events, shifts=(-1, 0, 1))) == 3
    assert len(st.ratio_sweep(prices, events)) == 4


def test_horizon_shape_and_caar_agree_at_the_pre_specified_horizons(planted):
    prices, events, _ = planted
    panel = st.event_panel(prices, events, horizons=(5, 21), cost_bps=0.0,
                           bench_cost_bps=0.0, borrow_bps=0.0)
    shape = st.horizon_shape(panel, horizons=(5, 21))
    assert shape.loc[21, "mean_pct"] == pytest.approx(
        st.leg_stats(panel, "a_child_21_gross")["mean_pct"])
    # The synthetic child has no when-issued tail, so the CAAR starts at event time 0.
    caar = st.caar_path(prices, events, pre=0, post=21)
    assert caar.loc[0, "caar_pct"] == pytest.approx(0.0, abs=1e-9)
    assert len(caar) == 22
    assert (caar["n"] > 5).all()


# --------------------------------------------------------------------------- #
# The study's spine — the estimator is unbiased
# --------------------------------------------------------------------------- #
def test_planted_child_drift_is_recovered(planted):
    prices, events, truth = planted
    d = st.synthetic_detect(prices, events)
    assert d["child21_mean_pct"] > 0.5 * truth["planted_child_alpha_21d"] * 100
    assert d["child21_t"] > 2.0


def test_planted_parent_runup_is_recovered(planted):
    prices, events, _ = planted
    d = st.synthetic_detect(prices, events)
    assert d["parent_mean_pct"] > 0.0
    assert d["parent_t"] > 2.0


def test_null_stays_quiet(null_world):
    prices, events, _ = null_world
    d = st.synthetic_detect(prices, events)
    assert abs(d["child21_t"]) < 2.0
    assert abs(d["parent_t"]) < 2.0


def test_null_is_centred_across_seeds():
    ts = []
    for s in range(6):
        prices, events, _ = data.synthetic_daily(signal_strength=0.0, seed=930 + s)
        ts.append(st.synthetic_detect(prices, events)["child21_t"])
    ts = np.array(ts)
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 1


def test_detector_scales_with_signal_strength():
    means = []
    for ss in (0.0, 0.5, 1.0):
        prices, events, _ = data.synthetic_daily(signal_strength=ss, seed=930)
        means.append(st.synthetic_detect(prices, events)["child21_mean_pct"])
    assert means[0] < means[1] < means[2]
