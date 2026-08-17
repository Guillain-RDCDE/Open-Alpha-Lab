"""Weight-fitting, backtest invariants and the study's spine — all offline/synthetic.

The spine: the harness must recover the planted blend weight in *both* worlds, hand the
replication the higher vol-matched Sharpe when the fund's idiosyncratic risk is
uncompensated, and stay flat when that same risk is fairly paid. Costs and borrow only
ever reduce the replication's return; the month-end weight freeze prevents look-ahead.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hy_replication import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The fitted weight
# --------------------------------------------------------------------------- #
def test_rolling_beta_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=900)
    r_eq = pd.Series(rng.normal(0, 0.011, 900), index=idx)
    r_du = pd.Series(rng.normal(0, 0.004, 900), index=idx)
    r_hy = 0.40 * r_eq + 0.60 * r_du + pd.Series(rng.normal(0, 0.001, 900), index=idx)
    beta = st.rolling_beta(r_hy, r_eq, r_du, window=252).dropna()
    assert abs(beta.mean() - 0.40) < 0.02


def test_rolling_beta_warmup_is_nan():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-04", periods=400)
    a = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    c = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    beta = st.rolling_beta(a, b, c, window=252)
    assert beta.iloc[:251].isna().all()
    assert beta.iloc[252:].notna().all()


def test_held_out_weight_is_a_monthly_step_from_the_previous_month():
    idx = pd.bdate_range("2015-01-01", periods=400)
    beta = pd.Series(np.arange(400, dtype=float), index=idx)
    w = st.held_out_weights(beta)
    # constant within each calendar month
    assert w.groupby(w.index.to_period("M")).nunique().max() == 1
    # the first month has no predecessor to inherit from
    assert w[w.index.to_period("M") == pd.Period("2015-01")].isna().all()
    # February's weight is January's last value
    feb = w[w.index.to_period("M") == pd.Period("2015-02")].iloc[0]
    jan_last = beta[beta.index.to_period("M") == pd.Period("2015-01")].iloc[-1]
    assert feb == jan_last


def test_no_lookahead_perturbing_the_future_leaves_earlier_weights_alone():
    """Multiply the tail of the tape; weights before the perturbation must not move."""
    prices, _ = data.synthetic_panel(n_years=8, seed=954)
    r = prices.pct_change(fill_method=None)
    beta_a = st.rolling_beta(r["fund"], r["equity"], r["duration"], window=252)
    r2 = r.copy()
    r2.iloc[1200:, r2.columns.get_loc("fund")] *= 4.0
    beta_b = st.rolling_beta(r2["fund"], r2["equity"], r2["duration"], window=252)
    w_a = st.held_out_weights(beta_a).iloc[:1200]
    w_b = st.held_out_weights(beta_b).iloc[:1200]
    assert (w_a.fillna(-9) == w_b.fillna(-9)).all()


# --------------------------------------------------------------------------- #
# Backtest engine invariants
# --------------------------------------------------------------------------- #
def test_replicate_columns_and_no_nans(uncompensated):
    prices, _ = uncompensated
    bt = st.replicate(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    assert {"r_hy", "r_repl", "r_cash", "w", "turnover", "short_notional"}.issubset(bt.columns)
    assert not bt.isna().any().any()
    assert len(bt) > 1000


def test_replication_is_fully_funded_no_cash_created():
    """With zero friction the blend return is exactly w*equity + (1-w)*duration."""
    prices, _ = data.synthetic_panel(n_years=6, seed=954)
    bt = st.replicate(prices["fund"], prices["equity"], prices["duration"], prices["cash"],
                      cost_bps=0.0, borrow_bps_ann=0.0)
    r_eq = prices["equity"].pct_change(fill_method=None).reindex(bt.index)
    r_du = prices["duration"].pct_change(fill_method=None).reindex(bt.index)
    rebuilt = bt["w"] * r_eq + (1.0 - bt["w"]) * r_du
    assert (bt["r_repl"] - rebuilt).abs().max() < 1e-12


def test_costs_monotonically_reduce_the_replication(uncompensated):
    prices, _ = uncompensated
    means = []
    for c in (0.0, 5.0, 25.0):
        bt = st.replicate(prices["fund"], prices["equity"], prices["duration"],
                          prices["cash"], cost_bps=c)
        means.append(bt["r_repl"].mean())
    assert means[0] >= means[1] >= means[2]


def test_borrow_charge_only_bites_when_a_leg_goes_short():
    """No short notional -> the borrow rate is inert; that is why the sweep is flat."""
    prices, _ = data.synthetic_panel(n_years=8, seed=954)
    a = st.replicate(prices["fund"], prices["equity"], prices["duration"], prices["cash"],
                     cost_bps=0.0, borrow_bps_ann=0.0)
    b = st.replicate(prices["fund"], prices["equity"], prices["duration"], prices["cash"],
                     cost_bps=0.0, borrow_bps_ann=1000.0)
    if a["short_notional"].max() == 0.0:
        assert (a["r_repl"] - b["r_repl"]).abs().max() < 1e-15
    else:
        assert b["r_repl"].mean() < a["r_repl"].mean()


def test_borrow_charge_bites_on_a_levered_weight():
    """A fund that IS levered equity must fit w > 1 and pay borrow *inside* the engine.

    The real tape never goes short, so the borrow charge is inert there — which is only
    an honest thing to report if the charge demonstrably lands when a weight does go
    short. Build a tape whose fund is 1.4 x equity - 0.4 x duration and run it through
    :func:`replicate` end to end.
    """
    prices, _ = data.synthetic_panel(n_years=8, seed=954)
    r_eq = prices["equity"].pct_change(fill_method=None).fillna(0.0)
    r_du = prices["duration"].pct_change(fill_method=None).fillna(0.0)
    levered = 100.0 * (1.0 + 1.4 * r_eq - 0.4 * r_du).cumprod()

    free = st.replicate(levered, prices["equity"], prices["duration"], prices["cash"],
                        cost_bps=0.0, borrow_bps_ann=0.0)
    charged = st.replicate(levered, prices["equity"], prices["duration"], prices["cash"],
                           cost_bps=0.0, borrow_bps_ann=1000.0)

    assert free["w"].mean() > 1.0                      # the duration leg really is short
    short = float(free["short_notional"].mean())
    assert short == pytest.approx(free["w"].mean() - 1.0, abs=1e-9)
    # 1000 bps/yr on the short notional, charged daily, is exactly what disappears.
    lost = float(free["r_repl"].mean() - charged["r_repl"].mean())
    assert lost == pytest.approx(short * 0.10 / st.TRADING_DAYS, rel=1e-6)
    assert lost > 0.0


def test_turnover_is_small_and_monthly(uncompensated):
    prices, _ = uncompensated
    bt = st.replicate(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    changes = (bt["w"].diff().abs() > 0).sum()
    assert changes <= len(bt) / 15  # at most one change per ~15 trading days


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_vol_matched_diff_is_scale_invariant(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    d1 = st.vol_matched_diff(cmp["e_hy"], cmp["e_repl"])
    d2 = st.vol_matched_diff(cmp["e_hy"] * 3.0, cmp["e_repl"] * 0.5)
    assert np.allclose(d1.to_numpy(), d2.to_numpy())


def test_sharpe_diff_tstat_sign_matches_the_gap(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    assert np.sign(cmp["t_gap"]) == np.sign(cmp["excess_sharpe_gap"])


def test_welch_and_one_sample_finite(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    assert np.isfinite(st.one_sample_t(cmp["resid"].to_numpy()))
    assert np.isfinite(st.welch_t(cmp["e_hy"].to_numpy(), cmp["e_repl"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_ci_brackets_point(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    ci = st.bootstrap_sharpe_ci(cmp["e_repl"], n_boot=400, seed=954)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_bootstrap_ci_handles_a_too_short_series():
    ci = st.bootstrap_sharpe_ci(pd.Series([0.01, -0.01, 0.02]), n_boot=50)
    assert np.isnan(ci["sharpe"]) and ci["n_boot"] == 0


# --------------------------------------------------------------------------- #
# The study's spine — machinery is unbiased
# --------------------------------------------------------------------------- #
def test_weight_recovered_in_both_worlds(uncompensated, fairly_paid):
    """The blend weight must be measured correctly whether or not the fund loses."""
    for prices, truth in (uncompensated, fairly_paid):
        d = st.synthetic_detect(prices)
        assert abs(d["w_mean"] - truth["w_true"]) < 0.05


def test_uncompensated_risk_hands_the_win_to_the_replication(uncompensated):
    prices, truth = uncompensated
    d = st.synthetic_detect(prices)
    assert d["excess_sharpe_gap"] < -0.15          # replication ahead on matched vol
    assert d["residual_ann"] < -0.5 * truth["resid_drag"]
    assert d["t_gap"] < -1.5


def test_fairly_paid_risk_is_a_dead_heat(fairly_paid):
    """The null: same extra risk, fairly compensated -> no vol-matched advantage."""
    prices, _ = fairly_paid
    d = st.synthetic_detect(prices)
    assert abs(d["excess_sharpe_gap"]) < 0.30
    assert abs(d["t_gap"]) < 2.0


def test_null_across_seeds_is_centered():
    gaps = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=954 + s)[0])["excess_sharpe_gap"]
        for s in range(6)
    ])
    assert abs(gaps.mean()) < 0.20
    assert (np.abs(gaps) >= 0.35).sum() <= 1


def test_planted_effect_fires_across_seeds():
    gaps = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=954 + s)[0])["excess_sharpe_gap"]
        for s in range(6)
    ])
    assert gaps.mean() < -0.25
    assert (gaps < 0).all()


def test_replication_never_explains_everything(uncompensated):
    """By construction the fund carries idiosyncratic risk, so R-squared must be < 1."""
    prices, _ = uncompensated
    d = st.synthetic_detect(prices)
    assert 0.3 < d["r2"] < 0.95


# --------------------------------------------------------------------------- #
# Robustness helpers
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    eras = st.era_cut(cmp, split="2017-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["excess_sharpe_gap"]) and np.isfinite(e["t_gap"])


def test_crisis_table_shape(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    tbl = st.crisis_table(cmp)
    assert {"dd_hy", "dd_repl", "ret_hy", "ret_repl"}.issubset(tbl.columns)
    assert (tbl["dd_hy"] <= 0).all() and (tbl["dd_repl"] <= 0).all()


def test_cost_sweep_is_monotone_in_cost(uncompensated):
    prices, _ = uncompensated
    sw = st.cost_sweep(prices["fund"], prices["equity"], prices["duration"], prices["cash"],
                       grid=(0.0, 5.0, 25.0))
    assert sw["sharpe_repl"].is_monotonic_decreasing
    assert sw["excess_sharpe_gap"].is_monotonic_increasing  # gap moves toward the fund


def test_window_sweep_covers_the_grid(uncompensated):
    prices, _ = uncompensated
    sw = st.window_sweep(prices["fund"], prices["equity"], prices["duration"],
                         prices["cash"], grid=(252, 504))
    assert list(sw.index) == [252, 504]
    assert (sw["n_days"] > 500).all()
    assert sw.loc[504, "n_days"] < sw.loc[252, "n_days"]  # longer fit starts later


def test_leg_sweep_runs_over_alternative_duration_legs(uncompensated):
    """The duration leg is a design choice; the sweep must re-fit w to each of them."""
    prices, truth = uncompensated
    # A second, shorter-duration Treasury leg: same shocks, half the volatility.
    short_dur = 100.0 * (1.0 + prices["duration"].pct_change(fill_method=None).fillna(0.0) * 0.5).cumprod()
    sw = st.leg_sweep(prices["fund"], prices["equity"],
                      {"long": prices["duration"], "short": short_dur}, prices["cash"])
    assert list(sw.index) == ["long", "short"]
    assert sw["excess_sharpe_gap"].lt(0).all()          # planted give-up survives the leg
    # Halving the duration leg's vol must push the fitted equity share DOWN, because the
    # fund's own duration exposure now needs more of the (weaker) bond leg to match.
    assert sw.loc["short", "w_mean"] < sw.loc["long", "w_mean"]
    assert abs(sw.loc["long", "w_mean"] - truth["w_true"]) < 0.05


def test_lived_and_excess_cagr_are_reported_separately(uncompensated):
    """The Sharpe race is excess-of-cash; the CAGR a holder lived is not the same number."""
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    for lived, excess in ((cmp["cagr_hy_abs"], cmp["hy"]["cagr"]),
                          (cmp["cagr_repl_abs"], cmp["repl"]["cagr"])):
        assert lived > excess                                    # cash is positive here
        assert lived == pytest.approx(excess + cmp["cagr_cash_abs"], abs=0.01)


def test_r2_by_horizon_table(uncompensated):
    prices, _ = uncompensated
    cmp = st.compare(prices["fund"], prices["equity"], prices["duration"], prices["cash"])
    tbl = st.replication_r2_by_horizon(cmp["bt"])
    assert list(tbl.index) == ["D", "W", "ME", "QE"]
    assert ((tbl["r2"] > 0) & (tbl["r2"] < 1)).all()
