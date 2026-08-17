"""Measurement logic, backtest invariants and the study's spine — all offline/synthetic.

The spine has two halves, and the synthetic control has to police both:

- on the planted ladder the measured durations must **order with the planted WAM**
  and the distributed-lag pass-through must sum to ~1, while the rate-direction
  switch rule earns a positive gross excess (the planted rate path trends);
- on the null the duration *spread* collapses to ~0 and the switch rule finds
  nothing at |t| >= 2 across seeds.

Plus the usual invariants: one execution lag and no look-ahead, costs that only
ever reduce the net, and a placebo that matches the real rule's turnover.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cash_lag import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Realised-yield proxy
# --------------------------------------------------------------------------- #
def test_realised_yield_recovers_a_constant_accrual():
    """A NAV compounding at exactly 4%/yr must read back as ~4."""
    n = 400
    nav = pd.Series(100.0 * (1.0 + 0.04 / 252) ** np.arange(n),
                    index=pd.bdate_range("2018-01-01", periods=n))
    y = st.realised_yield(nav, window=21).dropna()
    # 4% simple accrual compounds to 4.08% annualised — the proxy reports the
    # compounded figure, which is what a holder actually earns.
    assert abs(y.mean() - 4.08) < 0.02
    assert y.std() < 1e-6


def test_realised_yield_warmup_is_nan(laddered):
    prices, truth = laddered
    y = st.realised_yield(prices[truth["names"][0]], window=21)
    assert y.iloc[:21].isna().all()
    assert y.iloc[21:].notna().all()


# --------------------------------------------------------------------------- #
# A. Measurement
# --------------------------------------------------------------------------- #
def test_effective_duration_recovers_the_planted_value(random_walk_ladder):
    """Under a random-walk rate the regression recovers the planted WAM/2 to ~10%.

    Measured on the random-walk fixture on purpose: when rate *changes* are
    autocorrelated (the trending fixture), OLS on the contemporaneous change also
    absorbs part of the correlated carry response and reads a little high. The
    unbiased recovery test therefore belongs on the unpredictable-rate world.
    """
    prices, truth = random_walk_ladder
    for v in truth["names"]:
        planted = truth["per_vehicle"][v]["duration_years"]
        got = st.effective_duration(prices[v], prices[data.RATE])["duration_years"]
        assert abs(got - planted) < max(0.1 * planted, 0.005)


def test_trending_rate_still_orders_the_durations_correctly(laddered):
    prices, truth = laddered
    for v in truth["names"]:
        planted = truth["per_vehicle"][v]["duration_years"]
        got = st.effective_duration(prices[v], prices[data.RATE])["duration_years"]
        assert 0.5 * planted < got < 2.0 * planted


def test_durations_order_with_planted_wam(laddered):
    prices, truth = laddered
    order = sorted(truth["names"], key=lambda v: truth["per_vehicle"][v]["wam_days"])
    got = [st.effective_duration(prices[v], prices[data.RATE])["duration_years"] for v in order]
    assert all(a < b for a, b in zip(got, got[1:]))


def test_duration_t_is_positive_for_a_real_duration(laddered):
    prices, truth = laddered
    d = st.effective_duration(prices[truth["slowest"]], prices[data.RATE])
    assert d["tstat"] > 4.0  # signed on the duration, not the raw slope


def test_null_has_no_duration_ordering(flat_panel):
    prices, truth = flat_panel
    durs = [st.effective_duration(prices[v], prices[data.RATE])["duration_years"]
            for v in truth["names"]]
    assert max(durs) - min(durs) < 0.01


def test_lag_profile_pass_through_sums_to_about_one(random_walk_ladder):
    """A bill portfolio eventually inherits the whole bill rate: sum(beta) ~ 1.

    Checked on the vehicles whose ladder fits inside the 63-day lag grid; the
    longest ladder's pass-through runs past the last regressor, so its sum is
    mechanically short and is not asserted here.
    """
    prices, truth = random_walk_ladder
    inside = [v for v in truth["names"] if truth["per_vehicle"][v]["wam_days"] <= 63]
    assert len(inside) >= 2
    for v in inside:
        p = st.lag_profile(prices[v], prices[data.RATE])
        assert 0.7 < p["sum_beta"] < 1.4
        assert p["n"] > 1000


def test_lag_profile_contemporaneous_is_the_duration_shock(laddered):
    """The long-WAM vehicle is marked down hardest on the day rates move."""
    prices, truth = laddered
    slow = st.lag_profile(prices[truth["slowest"]], prices[data.RATE])["contemporaneous"]
    fast = st.lag_profile(prices[truth["fastest"]], prices[data.RATE])["contemporaneous"]
    assert slow < fast


def test_centroid_lag_is_longer_for_a_longer_ladder(laddered):
    prices, truth = laddered
    slow = st.lag_profile(prices[truth["slowest"]], prices[data.RATE])["centroid_lag"]
    fast = st.lag_profile(prices[truth["fastest"]], prices[data.RATE])["centroid_lag"]
    assert slow > fast


def test_lag_window_sweep_shape(laddered):
    prices, truth = laddered
    rows = st.lag_window_sweep(prices[truth["slowest"]], prices[data.RATE], windows=(5, 21))
    assert [r["window"] for r in rows] == [5, 21]
    assert all(np.isfinite(r["centroid_lag"]) for r in rows)


# --------------------------------------------------------------------------- #
# B. Signal & backtest invariants
# --------------------------------------------------------------------------- #
def test_signal_is_binary_and_lagged(laddered):
    prices, _ = laddered
    sig = st.rate_direction_signal(prices[data.RATE], lookback=21)
    assert set(sig.dropna().unique().tolist()).issubset({-1.0, 1.0})
    assert sig.iloc[:22].isna().all()  # 21-day diff + the one-day execution shift


def test_signal_is_plus_one_when_rates_rise_monotonically():
    r = pd.Series(np.linspace(1.0, 5.0, 200), index=pd.bdate_range("2020-01-01", periods=200))
    assert st.rate_direction_signal(r, lookback=21).dropna().eq(1.0).all()


def test_signal_is_minus_one_when_rates_fall_monotonically():
    r = pd.Series(np.linspace(5.0, 1.0, 200), index=pd.bdate_range("2020-01-01", periods=200))
    assert st.rate_direction_signal(r, lookback=21).dropna().eq(-1.0).all()


def test_no_lookahead_signal_uses_only_the_past():
    """Perturbing the tail of the rate path must not move any earlier signal."""
    rng = np.random.default_rng(0)
    r = pd.Series(2.0 + np.cumsum(rng.normal(0, 0.02, 500)).clip(-1.5, None),
                  index=pd.bdate_range("2015-01-01", periods=500))
    a = st.rate_direction_signal(r)
    r2 = r.copy()
    r2.iloc[400:] += 3.0
    b = st.rate_direction_signal(r2)
    assert (a.iloc[:400].fillna(-9) == b.iloc[:400].fillna(-9)).all()


def test_backtest_holds_exactly_the_signalled_vehicle(laddered):
    prices, truth = laddered
    f, s = truth["fastest"], truth["slowest"]
    sig = st.rate_direction_signal(prices[data.RATE])
    bt = st.switch_backtest(prices, sig, fast=f, slow=s, bench=truth["names"][1], cost_bps=0.0)
    hold_fast = bt[bt["signal"] > 0]
    hold_slow = bt[bt["signal"] < 0]
    # only the switch days differ, and at cost 0 there are no differences at all
    assert (hold_fast["r_switch"] - hold_fast["r_fast"]).abs().max() < 1e-12
    assert (hold_slow["r_switch"] - hold_slow["r_slow"]).abs().max() < 1e-12


def test_backtest_columns_and_no_nans(laddered):
    prices, truth = laddered
    sig = st.rate_direction_signal(prices[data.RATE])
    bt = st.switch_backtest(prices, sig, fast=truth["fastest"], slow=truth["slowest"],
                            bench=truth["names"][1], cost_bps=2.0)
    assert {"r_fast", "r_slow", "r_bench", "signal", "r_switch", "excess"}.issubset(bt.columns)
    assert not bt.isna().any().any()


def test_costs_monotonically_reduce_the_net(laddered):
    prices, truth = laddered
    means = []
    for c in (0.0, 2.0, 10.0):
        ev = st.evaluate(prices, fast=truth["fastest"], slow=truth["slowest"],
                         bench=truth["names"][1], cost_bps=c)
        means.append(ev["switch"]["ann_bp"])
    assert means[0] > means[1] > means[2]


def test_placebo_matches_the_real_rule_turnover(laddered):
    prices, _ = laddered
    sig = st.rate_direction_signal(prices[data.RATE])
    real = int((sig.dropna().diff().abs().fillna(0.0) / 2).sum())
    pl = st.placebo_signal(sig, seed=1).dropna()
    got = int((pl.diff().abs().fillna(0.0) / 2).sum())
    assert abs(got - real) <= max(3, 0.15 * real)


def test_placebo_is_reproducible_and_seed_sensitive(laddered):
    prices, _ = laddered
    sig = st.rate_direction_signal(prices[data.RATE])
    a, b, c = (st.placebo_signal(sig, seed=s) for s in (4, 4, 5))
    assert (a.dropna() == b.dropna()).all()
    assert not (a.dropna() == c.dropna()).all()


def test_reversed_signal_is_the_negation(laddered):
    prices, _ = laddered
    sig = st.rate_direction_signal(prices[data.RATE])
    assert (st.reversed_signal(sig).dropna() == -sig.dropna()).all()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_a_planted_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 4000)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, 4000)
    b, se = st.hac_ols(y, x, lags=5)
    assert abs(b[1] - 2.0) < 0.05 and abs(b[0] - 0.5) < 0.05
    assert b[1] / se[1] > 10


def test_bootstrap_ci_brackets_the_point(laddered):
    prices, truth = laddered
    ev = st.evaluate(prices, fast=truth["fastest"], slow=truth["slowest"],
                     bench=truth["names"][1])
    ci = st.block_bootstrap_mean_ci(ev["excess"], n_boot=400, seed=923)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_one_sample_and_welch_are_finite(laddered):
    prices, truth = laddered
    ev = st.evaluate(prices, fast=truth["fastest"], slow=truth["slowest"],
                     bench=truth["names"][1])
    e = ev["excess"].to_numpy()
    assert np.isfinite(st.one_sample_t(e))
    assert np.isfinite(st.welch_t(e, ev["bt"]["r_fast"].to_numpy()))


# --------------------------------------------------------------------------- #
# The study's spine — the detector fires on the plant and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_ladder_shows_a_duration_spread(laddered):
    d = st.synthetic_detect(*laddered)
    assert d["duration_spread"] > 0.1


def test_planted_trend_makes_the_switch_rule_pay(flat_panel, laddered):
    d = st.synthetic_detect(*laddered)
    assert d["gross_bp"] > 20.0 and d["gross_t"] > 3.0


def test_ladder_alone_pays_without_any_rate_forecasting(random_walk_ladder):
    """The power floor: a real ladder pays even when rate moves are unpredictable.

    With ``trend_phi=0`` nothing about the next rate move is knowable, yet the
    switch rule still earns a small, reliably positive gross excess — because
    after a run-up the short ladder's book yield has already repriced while the
    long one's is stale, so rotating captures a *carry* difference rather than a
    forecast. This is the smallest effect the harness must be able to see, and it
    is far smaller than the trending case; the real tape comes in below even this.
    """
    d = st.synthetic_detect(*random_walk_ladder)
    assert d["duration_spread"] > 0.1
    assert 2.0 < d["gross_t"] < 12.0
    assert 0.0 < d["gross_bp"] < 40.0


def test_killing_the_ladder_kills_the_trade(random_walk_ladder, flat_panel):
    """The trade tracks the ladder, not the trend: no WAM spread, no gross edge."""
    with_ladder = st.synthetic_detect(*random_walk_ladder)
    without = st.synthetic_detect(*flat_panel)
    assert with_ladder["gross_bp"] > without["gross_bp"] + 5.0
    assert abs(without["gross_t"]) < 2.5


def test_null_kills_the_duration_ordering(flat_panel):
    d = st.synthetic_detect(*flat_panel)
    assert abs(d["duration_spread"]) < 0.02


def test_null_switch_rule_finds_nothing(flat_panel):
    d = st.synthetic_detect(*flat_panel)
    assert abs(d["gross_t"]) < 2.5


def test_null_across_seeds_is_centred():
    ts = np.array([st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=923 + s))["gross_t"]
                   for s in range(6)])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 2.0).sum() <= 1


# --------------------------------------------------------------------------- #
# Reporting harness
# --------------------------------------------------------------------------- #
def test_evaluate_reports_every_arm(laddered):
    prices, truth = laddered
    ev = st.evaluate(prices, fast=truth["fastest"], slow=truth["slowest"],
                     bench=truth["names"][1])
    for k in ("switch", "reversed", "placebo", "static_fast", "static_slow"):
        assert np.isfinite(ev[k]["ann_bp"]) and np.isfinite(ev[k]["tstat"])
    assert 0.0 < ev["frac_fast"] < 1.0
    assert ev["n_switches"] > 0


def test_lookback_and_cost_sweeps_shape(laddered):
    prices, truth = laddered
    kw = dict(fast=truth["fastest"], slow=truth["slowest"], bench=truth["names"][1])
    lb = st.lookback_sweep(prices, lookbacks=(21, 63), **kw)
    cs = st.cost_sweep(prices, cost_grid=(0.0, 2.0), **kw)
    assert [r["lookback"] for r in lb] == [21, 63]
    assert [r["cost_bps"] for r in cs] == [0.0, 2.0]
    assert cs[0]["ann_bp"] > cs[1]["ann_bp"]


def test_era_cut_returns_both_halves(laddered):
    prices, truth = laddered
    eras = st.era_cut(prices, split="2016-01-01", fast=truth["fastest"],
                      slow=truth["slowest"], bench=truth["names"][1])
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gross_bp"]) and np.isfinite(e["net_t"])


def test_duration_table_shape(laddered):
    prices, truth = laddered
    tbl = st.duration_table(prices, truth["names"])
    assert list(tbl.index) == list(truth["names"])
    assert {"duration_years", "tstat", "n"}.issubset(tbl.columns)


def test_lag_profile_degrades_gracefully_on_a_short_series():
    n = 60
    nav = pd.Series(100.0 * (1.0 + 0.03 / 252) ** np.arange(n),
                    index=pd.bdate_range("2020-01-01", periods=n))
    rate = pd.Series(np.linspace(1.0, 2.0, n), index=nav.index)
    p = st.lag_profile(nav, rate)
    assert np.isnan(p["sum_beta"])


@pytest.mark.parametrize("window", [5, 10, 21])
def test_realised_yield_windows_all_produce_finite_profiles(laddered, window):
    prices, truth = laddered
    p = st.lag_profile(prices[truth["slowest"]], prices[data.RATE], window=window)
    assert np.isfinite(p["sum_beta"])
