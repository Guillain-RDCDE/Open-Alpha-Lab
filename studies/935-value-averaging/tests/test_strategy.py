"""Engine invariants, inference primitives and the study's spine — all offline/synthetic.

The spine: on a planted mean-reverting tape an **exposure-matched** value-averaging
programme finishes reliably richer than dollar-cost averaging; on a zero-volatility
tape (nothing to trade against) the same comparison is flat. In between, a plain
drifting random walk hands VA only the small mechanical rebalancing bonus — much
less than the planted wobble — which is why the real-tape residual is read as noise
rather than as timing skill.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value_avg import data, strategy as st  # noqa: E402


def _dates(prices, n_months=36, start=0):
    me = data.month_ends(prices.index)
    idx = prices.index
    pos = {d: i for i, d in enumerate(idx)}
    dec = list(me[start: start + n_months + 1])
    exe = [idx[pos[d] + 1] for d in dec]
    return dec, exe


# --------------------------------------------------------------------------- #
# The value path
# --------------------------------------------------------------------------- #
def test_value_path_linear_at_zero_growth():
    v = st.value_path(12, contrib=1.0, growth_ann=0.0)
    assert np.allclose(v, np.arange(1, 13, dtype=float))


def test_value_path_grows_with_the_assumed_rate():
    v0 = st.value_path(36, growth_ann=0.0)
    v8 = st.value_path(36, growth_ann=0.08)
    assert v8[-1] > v0[-1]
    assert v8[0] == pytest.approx(v0[0], abs=1e-12)  # first month is one contribution either way
    assert np.all(np.diff(v8) > 0)


# --------------------------------------------------------------------------- #
# Engine invariants
# --------------------------------------------------------------------------- #
def test_dca_invests_exactly_the_contribution(random_walk):
    prices, _ = random_walk
    dec, exe = _dates(prices)
    res = st.run_plan(prices["asset"], prices["cash"], dec, exe, mode="dca",
                      buffer_mult=6.0, cost_bps=0.0)
    assert res["n_trades"] == 36
    assert res["traded_notional"] == pytest.approx(36.0, rel=1e-9)
    assert res["bind_months"] == 0


def test_cash_benchmark_never_touches_the_asset(random_walk):
    prices, _ = random_walk
    dec, exe = _dates(prices)
    res = st.run_plan(prices["asset"], prices["cash"], dec, exe, mode="cash", buffer_mult=6.0)
    assert res["n_trades"] == 0
    assert res["terminal_equity"] == 0.0
    assert res["terminal_total"] > res["committed"]  # the cash leg accrues


def test_va_tracks_its_value_path(random_walk):
    """After each trade the equity sleeve should sit close to the target value."""
    prices, _ = random_walk
    dec, exe = _dates(prices)
    res = st.run_plan(prices["asset"], prices["cash"], dec, exe, mode="va",
                      buffer_mult=24.0, cost_bps=0.0)
    target_end = st.value_path(36)[-1]
    # One execution lag means a one-day price move of slack, not a free ride.
    assert res["terminal_equity"] == pytest.approx(target_end, rel=0.05)
    assert res["bind_months"] == 0  # a 24-month buffer is never exhausted here


def test_both_arms_commit_identical_capital(random_walk):
    prices, _ = random_walk
    dec, exe = _dates(prices)
    out = st.compare_window(prices["asset"], prices["cash"], dec, exe, buffer_mult=6.0)
    assert out["committed"] == pytest.approx(6.0 + 36.0)
    assert out["contributions"] == pytest.approx(36.0)


def test_zero_buffer_makes_the_cap_bind(wobbly):
    """With no buffer, value averaging cannot fund its own purchases in a falling tape."""
    prices, _ = wobbly
    df0 = st.rolling_race(prices["asset"], prices["cash"], 36, buffer_mult=0.0)
    df24 = st.rolling_race(prices["asset"], prices["cash"], 36, buffer_mult=24.0)
    assert (df0["va_bind_months"] > 0).mean() > (df24["va_bind_months"] > 0).mean()
    assert (df24["va_bind_months"] > 0).mean() == 0.0


def test_single_month_call_is_not_the_programme_total(wobbly):
    """The worst *single-month* cash call and the *programme-total* shortfall differ.

    A programme can bind in several months, so summing its unfunded calls and then
    quoting that sum as "the worst single month" overstates the cash a saver actually
    had to find at once. The first draft of this study's write-up did exactly that;
    this test makes the two quantities impossible to conflate silently.
    """
    prices, _ = wobbly
    df = st.rolling_race(prices["asset"], prices["cash"], 36, buffer_mult=0.0)
    s = st.summarise(df, 36)
    assert s["bind_months_total"] > s["n_windows"]                      # multi-month binds exist
    assert (df["va_shortfall_max_month"] <= df["va_shortfall"] + 1e-12).all()
    assert (df["va_shortfall_max_month"] < df["va_shortfall"] - 1e-9).any()
    assert s["worst_month_shortfall"] < s["worst_prog_shortfall"]


def test_costs_reduce_terminal_wealth(random_walk):
    prices, _ = random_walk
    dec, exe = _dates(prices)
    w = [st.run_plan(prices["asset"], prices["cash"], dec, exe, mode="va",
                     buffer_mult=12.0, cost_bps=c)["terminal_total"]
         for c in (0.0, 5.0, 50.0)]
    assert w[0] >= w[1] >= w[2]


def test_execution_lag_is_exactly_one_day(random_walk):
    """Perturbing prices strictly after a fill date cannot change that fill."""
    prices, _ = random_walk
    dec, exe = _dates(prices, n_months=12)
    base = st.run_plan(prices["asset"], prices["cash"], dec, exe, mode="va", buffer_mult=24.0)
    px2 = prices.copy()
    cut = px2.index.get_loc(exe[12])
    px2.iloc[cut + 1:, px2.columns.get_loc("asset")] *= 4.0
    pert = st.run_plan(px2["asset"], px2["cash"], dec, exe, mode="va", buffer_mult=24.0)
    assert pert["shares"] == pytest.approx(base["shares"], rel=1e-12)


def test_windows_never_run_past_the_tape(random_walk):
    prices, _ = random_walk
    me = data.month_ends(prices.index)
    wins = st.build_windows(prices.index, me, horizon_months=36)
    for dec, exe in wins:
        assert exe[-1] <= prices.index[-1]
        for d, e in zip(dec, exe):
            assert e > d


# --------------------------------------------------------------------------- #
# IRR & inference primitives
# --------------------------------------------------------------------------- #
def test_irr_recovers_a_known_rate():
    t0 = pd.Timestamp("2010-01-01")
    flows = [(t0, -100.0), (t0 + pd.Timedelta(days=365), 110.0)]
    assert st.irr_annual(flows) == pytest.approx(0.10, abs=2e-3)


def test_irr_is_nan_when_flows_do_not_bracket():
    t0 = pd.Timestamp("2010-01-01")
    assert np.isnan(st.irr_annual([(t0, -1.0), (t0 + pd.Timedelta(days=365), -1.0)]))


def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.5 + rng.normal(0, 1.0, 4000)
    assert st.newey_west_t(x, lags=12) > 3
    assert abs(st.newey_west_t(rng.normal(0, 1.0, 4000), lags=12)) < 3


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(29, 100)
    assert lo < 0.29 < hi


def test_bootstrap_ci_brackets_the_mean(random_walk):
    prices, _ = random_walk
    df = st.rolling_race(prices["asset"], prices["cash"], 36)
    ci = st.block_bootstrap_ci(df["gap_cents"], n_boot=400, block=36, seed=935)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_non_overlapping_subset_is_thinner(random_walk):
    prices, _ = random_walk
    df = st.rolling_race(prices["asset"], prices["cash"], 36)
    nov = st.non_overlapping_t(df, 36)
    assert 0 < nov["n"] <= len(df) // 36 + 1


# --------------------------------------------------------------------------- #
# The study's spine — machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_wobble_is_recovered(wobbly):
    """Exposure-matched, value averaging must win big on a strongly wobbling tape."""
    prices, _ = wobbly
    em = st.exposure_matched_race(prices["asset"], prices["cash"], 36,
                                  tol=0.01, max_iter=6, buffer_mult=6.0, cost_bps=1.0)
    assert abs(em["exposure_gap"]) < 0.02
    assert em["gap_mean_cents"] > 3.0
    assert em["t_hac"] > 2.0


def test_deterministic_tape_is_quiet(deterministic):
    """With zero volatility there is nothing to buy low and sell high — no gap."""
    prices, _ = deterministic
    em = st.exposure_matched_race(prices["asset"], prices["cash"], 36,
                                  tol=0.005, max_iter=8, buffer_mult=6.0, cost_bps=1.0)
    assert abs(em["gap_mean_cents"]) < 0.5


def test_random_walk_bonus_is_far_below_the_planted_effect(wobbly, random_walk):
    """A pure random walk yields only the small mechanical rebalancing bonus."""
    planted = st.exposure_matched_race(wobbly[0]["asset"], wobbly[0]["cash"], 36,
                                       tol=0.01, max_iter=6, buffer_mult=6.0, cost_bps=1.0)
    null = st.exposure_matched_race(random_walk[0]["asset"], random_walk[0]["cash"], 36,
                                    tol=0.01, max_iter=6, buffer_mult=6.0, cost_bps=1.0)
    assert planted["gap_mean_cents"] > 3.0 * null["gap_mean_cents"]


def test_exposure_gap_drives_the_raw_gap(random_walk):
    """The raw (unmatched) VA-minus-DCA gap tracks the average equity weight.

    This is the study's mechanism: on a rising tape a value path that grows more
    slowly than the market parks capital in cash, and the wealth gap follows the
    exposure gap, not the timing.
    """
    prices, _ = random_walk
    rows = []
    for g in (0.0, 0.06, 0.12):
        s = st.summarise(st.rolling_race(prices["asset"], prices["cash"], 36, growth_ann=g), 36)
        rows.append((s["va_invested_frac"] - s["dca_invested_frac"], s["gap_mean_cents"]))
    exposure = [r[0] for r in rows]
    gap = [r[1] for r in rows]
    assert exposure[0] < exposure[1] < exposure[2]
    assert gap[0] < gap[1] < gap[2]


def test_era_cut_returns_both_halves(random_walk):
    prices, _ = random_walk
    df = st.rolling_race(prices["asset"], prices["cash"], 36)
    eras = st.era_cut(df, split=str(df.index[len(df) // 2].date()), horizon_months=36)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gap_mean_cents"])


def test_sweep_tabulates_one_knob(random_walk):
    prices, _ = random_walk
    tbl = st.sweep(prices["asset"], prices["cash"], "cost_bps", (0.0, 25.0), horizon_months=36)
    assert list(tbl.index) == [0.0, 25.0]
    assert {"gap_mean_cents", "t_hac", "va_win_rate"}.issubset(tbl.columns)
