"""Policy logic, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a planted mean-reverting panel the 5/25 band rule beats the annual
calendar rule (it trades when dispersion is actually large); on the random-walk null
it does not. Around that, the invariants that make the race honest — exactly one
execution lag, costs that only ever subtract, a band rule that resets to target, and
a drift book that never trades.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebal_bands import data, strategy as st  # noqa: E402


def _rets(prices):
    cols = [c for c in prices.columns if c != "cash"]
    return prices[cols].pct_change().dropna()


def _cash(prices, rets):
    return prices["cash"].pct_change().reindex(rets.index)


# --------------------------------------------------------------------------- #
# Band arithmetic
# --------------------------------------------------------------------------- #
def test_band_thresholds_take_the_binding_side():
    thr = st.band_thresholds(np.array([0.60, 0.30, 0.10]))
    # 60% -> absolute 5pp binds; 30% -> absolute 5pp binds (25% of 30 = 7.5pp);
    # 10% -> relative 25% binds (2.5pp).
    assert np.isclose(thr[0], 0.05)
    assert np.isclose(thr[1], 0.05)
    assert np.isclose(thr[2], 0.025)


def test_band_breached_boundaries():
    tgt = np.array([0.6, 0.4])
    assert not st.band_breached(np.array([0.64, 0.36]), tgt)
    assert st.band_breached(np.array([0.65, 0.35]), tgt)
    assert st.band_breached(np.array([0.55, 0.45]), tgt)


def test_calendar_flags_mark_last_session_of_period():
    idx = pd.bdate_range("2019-12-27", "2020-01-08")
    yearly = st.calendar_flags(idx, "Y")
    assert idx[yearly][0] == pd.Timestamp("2019-12-31")
    assert yearly.sum() == 1
    q = st.calendar_flags(pd.bdate_range("2020-01-01", "2020-12-31"), "Q")
    assert q.sum() == 3  # the last quarter's end is the end of the sample -> no flag


def test_calendar_flags_reject_unknown_freq():
    with pytest.raises(ValueError):
        st.calendar_flags(pd.bdate_range("2020-01-01", periods=10), "W")


# --------------------------------------------------------------------------- #
# Engine invariants
# --------------------------------------------------------------------------- #
def test_drift_policy_never_trades(reverting):
    prices, _ = reverting
    r = st.run_policy(_rets(prices), (0.5, 0.5), policy="drift", cost_bps=25.0)
    assert r["n_rebals"] == 0
    assert r["traded_total"] == 0.0
    assert np.allclose(r["r"].to_numpy(), r["r_gross"].to_numpy())


def test_bad_target_raises(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    with pytest.raises(ValueError):
        st.run_policy(rets, (0.6, 0.6))
    with pytest.raises(ValueError):
        st.run_policy(rets, (0.3, 0.3, 0.4))
    with pytest.raises(ValueError):
        st.run_policy(rets, (0.5, 0.5), policy="fortnightly")


def test_nan_returns_rejected(reverting):
    prices, _ = reverting
    rets = _rets(prices).copy()
    rets.iloc[10, 0] = np.nan
    with pytest.raises(ValueError):
        st.run_policy(rets, (0.5, 0.5))


def test_costs_only_ever_subtract(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    means = []
    for c in (0.0, 5.0, 25.0, 100.0):
        r = st.run_policy(rets, (0.5, 0.5), policy="bands", cost_bps=c)
        means.append(r["r"].mean())
    assert means[0] >= means[1] >= means[2] >= means[3]


def test_cost_is_one_way_times_traded_notional(reverting):
    """The charged drag must equal cost_bps x traded notional, exactly."""
    prices, _ = reverting
    rets = _rets(prices)
    r = st.run_policy(rets, (0.5, 0.5), policy="quarterly", cost_bps=10.0)
    drag = (r["r_gross"] - r["r"]).sum()
    assert np.isclose(drag, 10.0 * 1e-4 * r["traded_total"])


def test_gross_return_is_free_of_costs(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    a = st.run_policy(rets, (0.5, 0.5), policy="bands", cost_bps=0.0)
    b = st.run_policy(rets, (0.5, 0.5), policy="bands", cost_bps=50.0)
    assert np.allclose(a["r_gross"].to_numpy(), b["r_gross"].to_numpy())


def test_execution_lag_is_exactly_one_day():
    """A trade decided at the close of t must print at the close of t+1, not at t.

    Two sleeves, flat until a single huge one-day move on bar 5 that blows the band.
    The rebalance must be recorded on bar 6.
    """
    n = 12
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n)}, index=idx)
    r.iloc[5, 0] = 0.50   # sleeve a jumps 50% -> weights go far outside the band
    out = st.run_policy(r, (0.5, 0.5), policy="bands", cost_bps=10.0)
    traded = out["traded"]
    assert traded.iloc[5] == 0.0            # not on the trigger day
    assert traded.iloc[6] > 0.0             # on the very next day
    assert out["n_rebals"] == 1


def test_weights_are_reset_to_target_after_a_band_trade():
    n = 12
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = pd.DataFrame({"a": np.zeros(n), "b": np.zeros(n)}, index=idx)
    r.iloc[5, 0] = 0.50
    out = st.run_policy(r, (0.5, 0.5), policy="bands", cost_bps=0.0)
    # The weight applied to bar 7 (start-of-day) is the post-trade target.
    assert np.allclose(out["weights"].iloc[7].to_numpy(), [0.5, 0.5])


def test_calendar_policies_trade_the_expected_number_of_times(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    yearly = st.run_policy(rets, (0.5, 0.5), policy="annual")
    quarterly = st.run_policy(rets, (0.5, 0.5), policy="quarterly")
    n_years = len(rets) / st.TRADING_DAYS
    assert abs(yearly["n_rebals"] - (n_years - 1)) <= 1
    assert abs(quarterly["n_rebals"] - 4 * (n_years - 1)) <= 4
    assert quarterly["traded_per_year"] > yearly["traded_per_year"]


def test_rebalancing_keeps_weights_closer_to_target_than_drift(random_walk):
    """Weight discipline is what a schedule actually buys — check it on the null world.

    The comparison is on the *worst* excursion, not the average: in a mean-reverting
    world the do-nothing book is dragged back to target for free, so only a
    random-walk tape makes the discipline claim non-trivial.
    """
    prices, _ = random_walk
    rets = _rets(prices)
    dev = {}
    for p in ("drift", "annual", "quarterly", "bands"):
        w = st.run_policy(rets, (0.5, 0.5), policy=p)["weights"].iloc[:, 0]
        dev[p] = float((w - 0.5).abs().max())
    assert dev["drift"] > dev["annual"]
    assert dev["drift"] > dev["bands"]
    assert dev["bands"] < 0.08


def test_band_rule_bounds_the_weight_path():
    """A 5/25 band on a 50/50 book must keep the observed weight inside ~55%."""
    prices, _ = data.synthetic_panel(n_years=10, seed=936)
    rets = _rets(prices)
    w = st.run_policy(rets, (0.5, 0.5), policy="bands", cost_bps=0.0)["weights"].iloc[:, 0]
    # Weights are inspected at the close, so one day of drift beyond the band is
    # allowed before the trade prints; a generous 8pp bound still has to hold.
    assert w.max() < 0.58 and w.min() > 0.42


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_and_auto_lags():
    rng = np.random.default_rng(2)
    assert abs(st.one_sample_t(rng.normal(0, 1, 4000))) < 3
    assert st.auto_lags(5000) >= st.auto_lags(500) >= 1


def test_diff_tstat_is_zero_for_identical_series(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    r = st.run_policy(rets, (0.5, 0.5), policy="annual")["r"]
    assert np.isnan(st.diff_tstat(r, r)) or abs(st.diff_tstat(r, r)) < 1e-8


def test_summary_fields_and_drawdown_sign(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    s = st.summary(st.run_policy(rets, (0.5, 0.5), policy="annual")["r"])
    for k in ("n_days", "sharpe", "vol_ann", "cagr", "max_drawdown", "tstat"):
        assert k in s
    assert s["max_drawdown"] <= 0.0
    assert np.isfinite(s["sharpe"])


def test_bootstrap_ci_brackets_the_point(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    a = st.run_policy(rets, (0.5, 0.5), policy="bands")["r"]
    b = st.run_policy(rets, (0.5, 0.5), policy="annual")["r"]
    ci = st.bootstrap_diff_ci(a, b, n_boot=300, seed=936)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# The study's spine — the harness recovers a planted effect and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_reversion_band_rule_beats_the_calendar():
    prices, _ = data.synthetic_panel(n_years=25, signal_strength=1.0, seed=936)
    d = st.synthetic_detect(prices)
    assert d["diff_bands_annual"] > 0.02
    assert d["t_bands_annual"] >= 2.0


def test_null_random_walk_no_band_advantage():
    prices, _ = data.synthetic_panel(n_years=25, signal_strength=0.0, seed=936)
    d = st.synthetic_detect(prices)
    assert abs(d["diff_bands_annual"]) < 0.02
    assert abs(d["t_bands_annual"]) < 2.0


def test_null_is_quiet_across_seeds():
    ts = np.array([
        st.synthetic_detect(data.synthetic_panel(n_years=25, signal_strength=0.0,
                                                 seed=936 + s)[0])["t_bands_annual"]
        for s in range(4)
    ])
    assert (np.abs(ts) >= 2.0).sum() == 0


def test_race_reports_every_policy(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    r = st.race(rets, (0.5, 0.5), cash=_cash(prices, rets), cost_bps=5.0)
    assert set(r["stats"]) == set(st.POLICIES)
    for key in ("bands_vs_annual", "bands_vs_quarterly", "quarterly_vs_annual"):
        assert np.isfinite(r["pairs"][key]["t_diff"])
    assert r["cost_drag_bps"]["drift"] == 0.0
    assert r["cost_drag_bps"]["quarterly"] > r["cost_drag_bps"]["annual"] > 0.0


def test_race_excess_of_cash_lowers_the_mean(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    cash = _cash(prices, rets)
    with_cash = st.race(rets, (0.5, 0.5), cash=cash)["stats"]["annual"]["mean_daily_bps"]
    without = st.race(rets, (0.5, 0.5), cash=None)["stats"]["annual"]["mean_daily_bps"]
    assert with_cash < without  # the cash leg has a positive yield


def test_race_rejects_a_misaligned_cash_leg(reverting):
    """A cash series that does not cover the sample must RAISE, not be zero-filled.

    Zero-filling missing cash days silently inflates every excess Sharpe in the table and
    leaves no trace in the output — exactly the kind of quiet upward bias this desk exists
    to catch.
    """
    prices, _ = reverting
    rets = _rets(prices)
    cash = _cash(prices, rets)
    with pytest.raises(ValueError):
        st.race(rets, (0.5, 0.5), cash=cash.iloc[: len(cash) // 2])
    # A handful of missing days is tolerated (holiday mismatches), and stays honest.
    nearly = cash.copy()
    nearly.iloc[:2] = np.nan
    assert np.isfinite(st.race(rets, (0.5, 0.5), cash=nearly)["stats"]["annual"]["sharpe"])


def test_sharpe_difference_is_NOT_cash_invariant_but_the_t_is(reverting):
    """The trap this study nearly fell into, pinned as a test.

    Subtracting the same cash leg from both arms cancels EXACTLY in the daily return
    difference — so the HAC t is identical — but a Sharpe is a nonlinear functional, so the
    Sharpe DIFFERENCE moves. Gross-of-cash and excess-of-cash Sharpe gaps may therefore
    never be quoted against each other.
    """
    prices, _ = reverting
    rets = _rets(prices)
    ex = st.race(rets, (0.5, 0.5), cash=_cash(prices, rets))["pairs"]["bands_vs_annual"]
    gr = st.race(rets, (0.5, 0.5), cash=None)["pairs"]["bands_vs_annual"]
    assert np.isclose(ex["t_diff"], gr["t_diff"])              # the t IS invariant
    assert not np.isclose(ex["sharpe_diff"], gr["sharpe_diff"])  # the Sharpe gap is NOT


def test_cost_sweep_is_monotone_in_cost(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    rows = st.cost_sweep(rets, (0.5, 0.5), cash=_cash(prices, rets), grid=(0.0, 10.0, 50.0))
    sharpes = [row["sharpe_bands"] for row in rows]
    assert sharpes[0] >= sharpes[1] >= sharpes[2]


def test_band_sweep_wider_bands_trade_less(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    rows = st.band_sweep(rets, (0.5, 0.5), cash=_cash(prices, rets),
                         grid=((0.02, 0.10), (0.05, 0.25), (0.10, 0.50)))
    traded = [row["traded_per_year"] for row in rows]
    assert traded[0] > traded[1] > traded[2]


def test_era_cut_returns_both_halves(reverting):
    prices, _ = reverting
    rets = _rets(prices)
    split = str(rets.index[len(rets) // 2].date())
    eras = st.era_cut(rets, (0.5, 0.5), cash=_cash(prices, rets), split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["diff_bands_annual"])


def test_era_cut_returns_none_on_a_stub(reverting):
    prices, _ = reverting
    rets = _rets(prices).iloc[:200]
    eras = st.era_cut(rets, (0.5, 0.5), split=str(rets.index[100].date()))
    assert eras["early"] is None and eras["late"] is None


def test_three_asset_book_runs():
    prices, _ = data.synthetic_panel(n_years=10, n_assets=3, seed=936)
    rets = _rets(prices)
    r = st.race(rets, (0.5, 0.3, 0.2), cash=_cash(prices, rets), cost_bps=5.0)
    assert np.isfinite(r["stats"]["bands"]["sharpe"])
    # The 20% sleeve gets the tighter relative band, so its weight stays close.
    w = r["runs"]["bands"]["weights"]["a2"]
    assert (w - 0.2).abs().max() < 0.06
