"""Event-study logic, execution-lag invariants, and the study's spine — all synthetic.

The spine: on a tape with a planted post-announcement shock the market-model event study
recovers it, with a placebo p-value near zero; on the null it recovers nothing and the
placebo p-value is unremarkable. Alongside that: the market model recovers a known beta,
the fast placebo kernel agrees with the slow per-event path to machine precision, the
tradable window never touches the day-0 return, and both frictions (cost, borrow) can
only reduce the net.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methodology_shock import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Market model
# --------------------------------------------------------------------------- #
def test_market_model_recovers_planted_beta(planted_pair):
    prices, _, truth = planted_pair
    r_t = prices["treated"].pct_change().to_numpy()[1:]
    r_c = prices["control"].pct_change().to_numpy()[1:]
    alpha, beta, sd = st.market_model(r_t, r_c)
    assert abs(beta - truth["beta"]) < 0.05
    assert sd > 0 and abs(alpha) < 1e-3


def test_market_model_refuses_a_short_window():
    a, b, sd = st.market_model(np.zeros(10), np.zeros(10))
    assert np.isnan(a) and np.isnan(b) and np.isnan(sd)


# --------------------------------------------------------------------------- #
# Event windows, execution lag, no look-ahead
# --------------------------------------------------------------------------- #
def test_event_window_length_matches_the_offsets(planted_pair):
    prices, events, _ = planted_pair
    res = st.event_abnormal(prices["treated"], prices["control"],
                            events[2]["announce"], window=(1, 10))
    assert res["ok"] and res["n_window"] == 10


def test_car_ignores_prices_after_the_window(planted_pair):
    """A (+1,+10) CAR must not move when the tape beyond day +10 is perturbed."""
    prices, events, _ = planted_pair
    d = events[3]["announce"]
    base = st.event_abnormal(prices["treated"], prices["control"], d, window=(1, 10))
    pos = int(prices.index.searchsorted(pd.Timestamp(d), side="left"))
    perturbed = prices.copy()
    perturbed.iloc[pos + 11:, perturbed.columns.get_loc("treated")] *= 3.0
    after = st.event_abnormal(perturbed["treated"], perturbed["control"], d, window=(1, 10))
    assert abs(base["car"] - after["car"]) < 1e-12


def test_estimation_window_excludes_the_event_window(planted_pair):
    """Alpha/beta come from a gapped pre-event window, so event-window prices cannot
    contaminate them: perturbing day +1 onwards leaves beta untouched."""
    prices, events, _ = planted_pair
    d = events[3]["announce"]
    base = st.event_abnormal(prices["treated"], prices["control"], d, window=(1, 10))
    pos = int(prices.index.searchsorted(pd.Timestamp(d), side="left"))
    perturbed = prices.copy()
    perturbed.iloc[pos + 1:, perturbed.columns.get_loc("treated")] *= 2.0
    after = st.event_abnormal(perturbed["treated"], perturbed["control"], d, window=(1, 10))
    assert abs(base["beta"] - after["beta"]) < 1e-12


def test_tradable_flag_marks_only_post_announcement_windows(null_pair):
    prices, events, _ = null_pair
    sw = st.window_sweep(prices, events, n_placebo=0)
    for _, r in sw.iterrows():
        lo = int(r["window"].split(",")[0].lstrip("["))
        assert r["tradable"] == (lo >= 1)


def test_pair_trade_never_earns_the_day_zero_return(planted_pair):
    """Entry is at the day-0 close, so the day-0 return (which carries the news) is
    outside the trade: perturbing the day -1 close cannot change the gross P&L."""
    prices, events, _ = planted_pair
    d = events[3]["announce"]
    evs = [dict(events[3])]
    base = st.pair_trade(prices, evs, hold_days=10, hedge="dollar",
                         cost_bps=0.0, borrow_bps_ann=0.0)
    pos = int(prices.index.searchsorted(pd.Timestamp(d), side="left"))
    perturbed = prices.copy()
    perturbed.iloc[pos - 1, perturbed.columns.get_loc("treated")] *= 1.05
    after = st.pair_trade(perturbed, evs, hold_days=10, hedge="dollar",
                          cost_bps=0.0, borrow_bps_ann=0.0)
    assert abs(base["mean_gross_bps"] - after["mean_gross_bps"]) < 1e-9


# --------------------------------------------------------------------------- #
# The fast placebo kernel must equal the slow per-event path
# --------------------------------------------------------------------------- #
def test_kernel_matches_event_abnormal(planted_pair):
    prices, events, _ = planted_pair
    k = st.PairKernel(prices["treated"], prices["control"])
    for window in [(1, 10), (-5, 5), (1, 21)]:
        for e in events[:4]:
            pos = int(k.index.searchsorted(pd.Timestamp(e["announce"]), side="left"))
            fast = float(k.car([pos], window)[0])
            slow = st.event_abnormal(prices["treated"], prices["control"],
                                     e["announce"], window=window)
            assert slow["ok"]
            assert abs(fast - slow["car"]) < 1e-12


def test_kernel_legal_positions_are_safe(planted_pair):
    prices, _, _ = planted_pair
    k = st.PairKernel(prices["treated"], prices["control"])
    for window in [(1, 10), (-10, 10), (1, 21)]:
        legal = k.legal_positions(window)
        assert legal.size > 100
        assert legal.min() - 21 - 250 >= 0
        assert legal.max() + window[1] + 1 <= k.n


# --------------------------------------------------------------------------- #
# Frictions
# --------------------------------------------------------------------------- #
def test_costs_monotonically_reduce_the_net(planted_pair):
    prices, events, _ = planted_pair
    nets = [st.pair_trade(prices, events, cost_bps=c, borrow_bps_ann=0.0)["mean_net_bps"]
            for c in (0.0, 5.0, 25.0)]
    assert nets[0] > nets[1] > nets[2]


def test_borrow_monotonically_reduces_the_net(planted_pair):
    prices, events, _ = planted_pair
    nets = [st.pair_trade(prices, events, cost_bps=0.0, borrow_bps_ann=b)["mean_net_bps"]
            for b in (0.0, 100.0, 500.0)]
    assert nets[0] > nets[1] > nets[2]


def test_gross_is_never_worse_than_net(planted_pair):
    prices, events, _ = planted_pair
    pt = st.pair_trade(prices, events, cost_bps=5.0, borrow_bps_ann=50.0)
    assert (pt["table"]["gross_bps"] > pt["table"]["net_bps"]).all()


def test_financing_is_charged_on_the_residual_dollar_exposure(planted_pair):
    """The beta-hedged pair is NOT dollar-neutral: (1 - beta) units of NAV are a real net
    position. Raising the bill rate must move the net by exactly that residual x rate x
    days, with the sign of (1 - beta) — a cost when net long, a credit when net short."""
    prices, events, _ = planted_pair
    a = st.pair_trade(prices, events, cost_bps=0.0, borrow_bps_ann=0.0,
                      finance_bps_ann=0.0, hedge="beta")
    b = st.pair_trade(prices, events, cost_bps=0.0, borrow_bps_ann=0.0,
                      finance_bps_ann=500.0, hedge="beta")
    assert (a["table"]["finance_bps_charged"] == 0.0).all()
    # beta here is ~1.15 (net SHORT dollars), so financing is a CREDIT: net goes up.
    assert (b["table"]["beta"] > 1.0).all()
    assert (b["table"]["finance_bps_charged"] < 0.0).all()
    assert b["mean_net_bps"] > a["mean_net_bps"]
    # And the naive dollar-neutral variant has no residual at all, at any rate.
    d = st.pair_trade(prices, events, cost_bps=0.0, borrow_bps_ann=0.0,
                      finance_bps_ann=500.0, hedge="dollar")
    assert (d["table"]["net_dollar_exposure"].abs() < 1e-12).all()
    assert (d["table"]["finance_bps_charged"].abs() < 1e-12).all()


def test_cost_sweep_covers_the_grid(planted_pair):
    prices, events, _ = planted_pair
    sw = st.cost_sweep(prices, events, cost_grid=(0.0, 5.0), borrow_grid=(0.0, 100.0))
    assert len(sw) == 4
    assert set(sw.columns) >= {"cost_bps", "borrow_bps_ann", "mean_net_bps", "t_net"}


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_wilson_interval_brackets_the_point():
    lo, hi = st.wilson_interval(3, 7)
    assert lo < 3 / 7 < hi


def test_bootstrap_ci_brackets_the_mean():
    ci = st.bootstrap_car_ci([-120.0, -157.0, 68.0, 192.0, -96.0, -22.0, -120.0], seed=919)
    assert ci["ci_low"] <= ci["mean_bps"] <= ci["ci_high"]


def test_block_bootstrap_ci_brackets_the_car():
    rng = np.random.default_rng(0)
    ci = st.block_bootstrap_ar_ci(rng.normal(0, 0.01, 500), window_len=10, block=5, seed=919)
    assert ci["ci_low"] <= ci["car_bps"] <= ci["ci_high"]


def test_minimum_detectable_car_scales_with_dispersion():
    tight = {"n_events": 7, "sd_car_bps": 20.0}
    loose = {"n_events": 7, "sd_car_bps": 200.0}
    assert st.minimum_detectable_car(loose) > st.minimum_detectable_car(tight) > 0


# --------------------------------------------------------------------------- #
# The study's spine — the detector is powered on a plant and quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_shock_is_recovered(planted_panel):
    pairs, events, truth = planted_panel
    s = st.run_event_study(pairs, events, window=(1, 10))
    assert s["n_events"] == truth["n_events"]
    assert s["mean_car_bps"] > 0.5 * truth["planted_car_bps"]
    assert s["t_cross_event"] > 2.0
    assert s["hit_rate"] > 0.7


def test_planted_shock_clears_the_placebo(planted_panel):
    pairs, events, _ = planted_panel
    pl = st.placebo_test(pairs, events, window=(1, 10), n_draws=400, seed=919)
    assert pl["p_two_sided"] < 0.05
    assert abs(pl["null_mean_bps"]) < 40.0


def test_null_is_quiet(null_panel):
    pairs, events, _ = null_panel
    d = st.synthetic_detect(pairs, events, window=(1, 10), n_draws=400)
    assert abs(d["mean_car_bps"]) < 150.0
    assert d["placebo_p"] > 0.05


def test_null_is_centred_across_seeds():
    means = np.array([
        st.run_event_study(*data.synthetic_panel(signal_strength=0.0, seed=919 + 7 * k)[:2],
                           window=(1, 10))["mean_car_bps"]
        for k in range(6)
    ])
    assert abs(means.mean()) < 90.0
    assert (np.abs(means) > 250.0).sum() == 0


def test_planted_shock_is_absent_before_the_announcement(planted_panel):
    """The plant starts at day +1, so a pre-announcement window must stay empty."""
    pairs, events, _ = planted_panel
    pre = st.run_event_study(pairs, events, window=(-10, -1))
    post = st.run_event_study(pairs, events, window=(1, 10))
    assert abs(pre["mean_car_bps"]) < 0.4 * abs(post["mean_car_bps"])


# --------------------------------------------------------------------------- #
# Robustness helpers
# --------------------------------------------------------------------------- #
def test_jackknife_drops_one_event_at_a_time(planted_panel):
    pairs, events, _ = planted_panel
    jk = st.jackknife(pairs, events, window=(1, 10))
    assert len(jk) == len(events)
    assert set(jk.columns) >= {"dropped", "own_car_bps", "mean_car_bps_without", "t_without"}


def test_era_cut_returns_both_halves(planted_pair):
    prices, events, _ = planted_pair
    eras = st.era_cut(prices, events, split=events[len(events) // 2]["announce"],
                      window=(1, 10))
    assert eras["early"] is not None and eras["late"] is not None
    assert eras["early"]["n_events"] + eras["late"]["n_events"] == len(events)


def test_deployed_capital_race_is_sparse(planted_pair):
    prices, events, _ = planted_pair
    dc = st.deployed_capital_race(prices, prices["cash"], events, hold_days=10)
    assert dc["n_live_events"] == len(events)
    assert dc["n_dropped_pre_cash"] == 0
    assert 0 < dc["n_live_days"] <= 10 * len(events)
    # The overlay is flat on the overwhelming majority of days.
    assert dc["n_live_days"] < 0.05 * dc["summary"]["n_days"]


def test_deployed_capital_race_drops_events_before_the_cash_series(planted_pair):
    """A short cash series must DROP the earlier events, never teleport them onto its
    first session. BIL starts in 2007; a naive searchsorted would stack every pre-2007
    event's P&L into the first ten days of the cash era and invent a drawdown."""
    prices, events, _ = planted_pair
    cutoff = prices.index[len(prices) // 2]
    short_cash = prices["cash"].loc[prices.index >= cutoff]
    dc = st.deployed_capital_race(prices, short_cash, events, hold_days=10)
    n_before = sum(1 for e in events if pd.Timestamp(e["announce"]) < cutoff)
    assert n_before > 0                      # the fixture really does straddle the cutoff
    assert dc["n_dropped_pre_cash"] == n_before
    assert dc["n_live_events"] == len(events) - n_before
    assert dc["n_live_days"] == 10 * dc["n_live_events"]   # no collisions on day one
    # Nothing was written into the first ten sessions of the cash era.
    assert (dc["overlay"].iloc[:11] == 0.0).all()


def test_window_sweep_reports_every_window(null_pair):
    prices, events, _ = null_pair
    sw = st.window_sweep(prices, events, n_placebo=0)
    assert len(sw) == len(st.DESCRIPTIVE_WINDOWS) + len(st.TRADABLE_WINDOWS)
    assert sw["n_events"].gt(0).all()
