"""Signal logic, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a world where the front end genuinely trends, the switch rule beats the
static intermediate-duration leg *and* an exposure-matched random control; on a
random-walk null it does neither. Costs monotonically reduce the strategy's return; the
single execution lag prevents look-ahead; and the random control's turnover handicap is
made explicit so a gross-vs-net comparison can be read honestly.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from front_end_trend import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Signal tests
# --------------------------------------------------------------------------- #
def test_signal_is_binary_and_lagged(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"], lookback=63)
    assert set(sig.dropna().unique().tolist()).issubset({0.0, 1.0})
    assert sig.iloc[:64].isna().all()   # 63-day window + the one-day execution shift


def test_signal_all_in_when_rates_fall_monotonically():
    y = pd.Series(np.linspace(5.0, 1.0, 400),
                  index=pd.bdate_range("2020-01-02", periods=400))
    assert st.rate_trend_signal(y, lookback=63).dropna().eq(1.0).all()


def test_signal_all_out_when_rates_rise_monotonically():
    y = pd.Series(np.linspace(1.0, 5.0, 400),
                  index=pd.bdate_range("2020-01-02", periods=400))
    assert st.rate_trend_signal(y, lookback=63).dropna().eq(0.0).all()


def test_exactly_one_execution_lag():
    """The signal acted on at day t must be computable from yields through day t-1."""
    rng = np.random.default_rng(0)
    y = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.02, 400)),
                  index=pd.bdate_range("2020-01-02", periods=400))
    sig = st.rate_trend_signal(y, lookback=63)
    raw = (st.rate_change(y, 63) < 0).astype(float)
    raw[st.rate_change(y, 63).isna()] = np.nan
    # sig[t] == raw[t-1]: one shift, no more, no less.
    assert (sig.iloc[1:].fillna(-9).to_numpy() == raw.iloc[:-1].fillna(-9).to_numpy()).all()


def test_no_lookahead_future_perturbation_leaves_past_signals_alone():
    rng = np.random.default_rng(1)
    y = pd.Series(3.0 + np.cumsum(rng.normal(0, 0.02, 600)),
                  index=pd.bdate_range("2015-01-01", periods=600))
    s1 = st.rate_trend_signal(y)
    y2 = y.copy()
    y2.iloc[500:] += 3.0            # perturb only the future
    s2 = st.rate_trend_signal(y2)
    assert (s1.iloc[:500].fillna(-9) == s2.iloc[:500].fillna(-9)).all()


def test_shy_momentum_signal_is_binary_lagged_and_causal():
    px = pd.Series(np.linspace(100, 120, 400),
                   index=pd.bdate_range("2020-01-02", periods=400))
    sig = st.shy_momentum_signal(px)
    assert set(sig.dropna().unique().tolist()).issubset({0.0, 1.0})
    assert sig.dropna().eq(1.0).all()             # a rising short-bond price = long duration
    assert sig.iloc[:253].isna().all()            # 252-day window + the execution shift


def test_random_control_matches_in_market_fraction(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    rand = st.random_signal(sig, seed=0)
    assert abs(float(sig.dropna().mean()) - float(rand.dropna().mean())) < 0.05


def test_random_control_reproducible_and_seed_sensitive(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    assert (st.random_signal(sig, seed=5).dropna()
            == st.random_signal(sig, seed=5).dropna()).all()
    assert not (st.random_signal(sig, seed=5).dropna()
                == st.random_signal(sig, seed=6).dropna()).all()


def test_random_control_churns_far_more_than_a_trend_rule(trending_rates):
    """The control's handicap, made explicit: a coin flip switches ~2p(1-p)N times.

    This is why the switch-vs-random comparison must be read GROSS as well as net —
    at any positive cost the random arm pays friction the rule never does.
    """
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    rand = st.random_signal(sig, seed=0)
    n_sig = float(sig.diff().abs().sum())
    n_rand = float(rand.diff().abs().sum())
    assert n_rand > 3.0 * n_sig


# --------------------------------------------------------------------------- #
# Backtest engine invariants
# --------------------------------------------------------------------------- #
def test_backtest_columns_and_no_nans(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    bt = st.run_backtest(frame["long"], frame["cash"], sig, cost_bps=2.0)
    assert {"r_long", "r_cash", "signal", "r_strategy"}.issubset(bt.columns)
    assert not bt.isna().any().any()


def test_out_days_earn_cash_not_duration():
    y = pd.Series(np.linspace(1.0, 5.0, 300),
                  index=pd.bdate_range("2020-01-02", periods=300))
    long_leg = pd.Series(100 * np.exp(np.cumsum(np.full(300, -0.001))), index=y.index)
    cash = pd.Series(100 * (1.0 + 0.03 / 252) ** np.arange(300), index=y.index)
    sig = st.rate_trend_signal(y)
    bt = st.run_backtest(long_leg, cash, sig, cost_bps=0.0)
    out = bt[bt["signal"] == 0.0]
    assert len(out) > 10
    assert ((out["r_strategy"] - out["r_cash"]).abs() < 1e-12).all()


def test_costs_monotonically_reduce_return(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    means = [st.run_backtest(frame["long"], frame["cash"], sig, cost_bps=c)["r_strategy"].mean()
             for c in (0.0, 2.0, 25.0)]
    assert means[0] >= means[1] >= means[2]


def test_cost_is_charged_only_on_switch_days(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    free = st.run_backtest(frame["long"], frame["cash"], sig, cost_bps=0.0)
    paid = st.run_backtest(frame["long"], frame["cash"], sig, cost_bps=10.0)
    gap = (free["r_strategy"] - paid["r_strategy"])
    n_flips = int(free["signal"].diff().abs().fillna(0.0).sum())
    assert (gap > 1e-12).sum() == n_flips
    assert np.allclose(gap[gap > 1e-12].to_numpy(), 10.0 * 1e-4)


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_one_sample_finite(trending_rates):
    frame, _ = trending_rates
    cmp = st.compare_rate_rule(frame["rate"], frame["long"], frame["mid"], frame["cash"])
    assert np.isfinite(cmp["t_diff_vs_mid"])
    assert np.isfinite(st.one_sample_t(cmp["e_switch"].to_numpy()))
    assert np.isfinite(st.welch_t(cmp["e_switch"].to_numpy(), cmp["e_mid"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_ci_brackets_point(trending_rates):
    frame, _ = trending_rates
    cmp = st.compare_rate_rule(frame["rate"], frame["long"], frame["mid"], frame["cash"])
    ci = st.bootstrap_sharpe_ci(cmp["e_mid"], n_boot=400, seed=925)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_bootstrap_diff_ci_brackets_point(trending_rates):
    frame, _ = trending_rates
    cmp = st.compare_rate_rule(frame["rate"], frame["long"], frame["mid"], frame["cash"])
    d = st.bootstrap_diff_ci(cmp["e_switch"], cmp["e_mid"], n_boot=400, seed=925)
    assert d["ci_low"] <= d["diff"] <= d["ci_high"]
    assert abs(d["diff"] - cmp["excess_sharpe_adv"]) < 1e-9


# --------------------------------------------------------------------------- #
# The study's spine — the detector fires on a planted effect, quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_rate_trend_is_recovered(trending_rates):
    frame, _ = trending_rates
    d = st.synthetic_detect(frame)
    assert d["excess_sharpe_adv"] > 0.4
    assert d["t_diff_vs_mid"] > 2.0


def test_planted_rate_trend_beats_the_random_control(trending_rates):
    frame, _ = trending_rates
    assert st.synthetic_detect(frame)["sharpe_adv_vs_random"] > 0.3


def test_planted_effect_recovered_across_seeds():
    advs = np.array([st.synthetic_detect(f)["excess_sharpe_adv"]
                     for f, _ in data.synthetic_panel(n_seeds=5, signal_strength=1.0)])
    assert (advs > 0.4).all()


def test_null_produces_no_advantage(random_walk_rates):
    frame, _ = random_walk_rates
    assert abs(st.synthetic_detect(frame)["excess_sharpe_adv"]) < 0.4


def test_null_is_centred_across_seeds():
    advs = np.array([st.synthetic_detect(f)["excess_sharpe_adv"]
                     for f, _ in data.synthetic_panel(n_seeds=8, signal_strength=0.0)])
    assert abs(advs.mean()) < 0.25
    assert (np.abs(advs) >= 0.5).sum() <= 1


def test_null_day_level_spread_is_noise(random_walk_rates):
    frame, _ = random_walk_rates
    cd = st.conditional_day_test(frame["rate"], frame["long"], frame["cash"])
    assert abs(cd["welch_t"]) < 2.0


def test_planted_day_level_spread_is_real(trending_rates):
    frame, _ = trending_rates
    cd = st.conditional_day_test(frame["rate"], frame["long"], frame["cash"])
    assert cd["spread_bp"] > 0
    assert cd["welch_t"] > 2.0


# --------------------------------------------------------------------------- #
# Robustness harness shape
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(trending_rates):
    frame, _ = trending_rates
    eras = st.era_cut(frame["rate"], frame["long"], frame["mid"], frame["cash"],
                      split="2014-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["excess_sharpe_adv"])


def test_lookback_and_cost_sweeps_have_expected_shape(trending_rates):
    frame, _ = trending_rates
    lb = st.lookback_sweep(frame["rate"], frame["long"], frame["mid"], frame["cash"],
                           lookbacks=(21, 63))
    assert [r["lookback"] for r in lb] == [21, 63]
    cs = st.timer_stats(frame["rate"], frame["long"], frame["mid"], frame["cash"],
                        cost_bps_grid=(0.0, 25.0))
    assert cs[0]["sharpe_switch"] >= cs[1]["sharpe_switch"]


def test_random_control_sweep_reports_spread(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    sw = st.random_control_sweep(frame["long"], frame["mid"], frame["cash"], sig, n_seeds=5)
    assert sw["adv_min"] <= sw["adv_mean"] <= sw["adv_max"]
    assert 0.0 <= sw["frac_t_ge_2"] <= 1.0


def test_constant_weight_control_holds_its_weight_with_drift_turnover_only(trending_rates):
    """The turnover-free exposure control: same average duration, no regime flips."""
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    w = float(sig.dropna().mean())
    ctrl = st.constant_weight_control(frame["long"], frame["cash"], w, cost_bps=2.0)
    assert abs(ctrl["weight"] - w) < 1e-12
    # daily drift-rebalancing turns over a few percent of NAV a year per unit of vol —
    # nothing like a coin flipping in and out of a whole position ~2p(1-p)N times.
    rand_turnover = float(st.random_signal(sig, seed=0).diff().abs().mean()) * st.TRADING_DAYS
    assert ctrl["turnover_ann"] < rand_turnover
    assert np.isfinite(ctrl["summary"]["sharpe"])


def test_matched_blend_race_fires_on_planted_trend(trending_rates):
    frame, _ = trending_rates
    sig = st.rate_trend_signal(frame["rate"])
    r = st.matched_blend_race(frame["long"], frame["mid"], frame["cash"], sig)
    assert abs(r["weight"] - float(sig.dropna().mean())) < 0.02
    assert r["adv_vs_blend"] > 0.4          # real timing beats matched constant exposure
    assert r["t_diff_vs_blend"] > 2.0


def test_matched_blend_race_is_quiet_on_the_null(random_walk_rates):
    frame, _ = random_walk_rates
    sig = st.rate_trend_signal(frame["rate"])
    r = st.matched_blend_race(frame["long"], frame["mid"], frame["cash"], sig)
    assert abs(r["adv_vs_blend"]) < 0.4


def test_conditional_day_test_reports_a_hac_spread_t(trending_rates, random_walk_rates):
    """Welch assumes iid draws; the HAC contrast t is the one the study reads."""
    planted, _ = trending_rates
    null, _ = random_walk_rates
    cp = st.conditional_day_test(planted["rate"], planted["long"], planted["cash"])
    cn = st.conditional_day_test(null["rate"], null["long"], null["cash"])
    assert cp["spread_hac_t"] > 2.0
    assert abs(cn["spread_hac_t"]) < 2.0
    # the HAC t must be the *weaker* claim on a persistent signal, not a free upgrade
    assert cp["spread_hac_t"] < cp["welch_t"]


def test_calendar_year_table_shape(trending_rates):
    frame, _ = trending_rates
    tbl = st.calendar_year_table(frame["rate"], frame["long"], frame["mid"], frame["cash"])
    assert {"switch_pct", "ief_pct", "tlt_pct", "in_duration_frac", "n_days"}.issubset(tbl.columns)
    assert len(tbl) > 5
