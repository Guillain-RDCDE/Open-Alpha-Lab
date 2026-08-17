"""Signal logic, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a panel with a *planted* stale-price catch-up the lagged HAC regression
recovers the planted beta and the top-decile next-day rule earns a positive trigger-day
return; on the null (identical contemporaneous correlation, no lagged link) both are
quiet. Costs and borrow only ever reduce a return; the single execution lag is real.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stale_nav import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_ols_hac_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    y = 0.5 * x + rng.normal(0, 0.5, 4000)
    reg = st.ols_hac(y, x)
    assert reg["beta"] == pytest.approx(0.5, abs=0.05)
    assert reg["t_beta"] > 10
    assert reg["n"] == 4000


def test_ols_hac_quiet_on_independent_series():
    rng = np.random.default_rng(1)
    reg = st.ols_hac(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000))
    assert abs(reg["beta"]) < 0.1
    assert abs(reg["t_beta"]) < 3


def test_ols_hac_handles_short_sample():
    reg = st.ols_hac(np.arange(5.0), np.arange(5.0))
    assert np.isnan(reg["beta"])


def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(2)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_welch_and_one_sample_finite(planted):
    prices, truth = planted
    r = prices.pct_change()
    e = (r["FUND1"] - r["cash"]).dropna()
    assert np.isfinite(st.one_sample_t(e.to_numpy()))
    assert np.isfinite(st.welch_t(e.to_numpy(), r["SPY"].dropna().to_numpy()))


# --------------------------------------------------------------------------- #
# Signal tests
# --------------------------------------------------------------------------- #
def test_trigger_signal_is_binary_with_warmup(planted):
    prices, _ = planted
    sig = st.trigger_signal(prices.pct_change()["SPY"])
    assert set(sig.dropna().unique().tolist()).issubset({0.0, 1.0})
    assert sig.iloc[:250].isna().all()


def test_trigger_frequency_matches_the_quantile(planted):
    prices, _ = planted
    sig = st.trigger_signal(prices.pct_change()["SPY"], top_q=0.90)
    assert 0.08 < float(sig.dropna().mean()) < 0.14


def test_trigger_threshold_is_monotone_in_q(planted):
    prices, _ = planted
    spy = prices.pct_change()["SPY"]
    f80 = float(st.trigger_signal(spy, top_q=0.80).dropna().mean())
    f95 = float(st.trigger_signal(spy, top_q=0.95).dropna().mean())
    assert f80 > f95


def test_no_lookahead_expanding_threshold(planted):
    """Perturbing only the future must leave earlier signals untouched (causality)."""
    prices, _ = planted
    spy = prices.pct_change()["SPY"]
    s1 = st.trigger_signal(spy)
    spy2 = spy.copy()
    spy2.iloc[3000:] *= 5.0
    s2 = st.trigger_signal(spy2)
    assert (s1.iloc[:3000].fillna(-9) == s2.iloc[:3000].fillna(-9)).all()


def test_exactly_one_execution_lag(planted):
    """The position on day t must be the signal of day t-1 — one lag, no more, no less."""
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    bt = st.run_rule(r["FUND1"], r["cash"], sig, cost_bps=0.0)
    lagged = sig.reindex(bt.index).shift(1).fillna(0.0)
    assert (bt["position"] == lagged.reindex(bt.index)).all()


def test_extra_lag_shifts_the_position(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    a = st.run_rule(r["FUND1"], r["cash"], sig, cost_bps=0.0)
    b = st.run_rule(r["FUND1"], r["cash"], sig, cost_bps=0.0, extra_lag=1)
    common = a.index.intersection(b.index)
    assert (b["position"].reindex(common)
            == a["position"].reindex(common).shift(1).fillna(0.0)).all()


# --------------------------------------------------------------------------- #
# Backtest engine invariants
# --------------------------------------------------------------------------- #
def test_backtest_columns_and_no_nans(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    bt = st.run_rule(r["FUND1"], r["cash"], sig)
    assert {"position", "r_fund", "r_cash", "r_strategy", "excess"}.issubset(bt.columns)
    assert not bt.isna().any().any()


def test_off_days_earn_exactly_cash(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    bt = st.run_rule(r["FUND1"], r["cash"], sig, cost_bps=0.0)
    off = bt[(bt["position"] == 0) & (bt["position"].shift(1).fillna(0.0) == 0)]
    assert (off["excess"].abs() < 1e-12).all()


def test_costs_monotonically_reduce_return(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    means = [st.run_rule(r["FUND1"], r["cash"], sig, cost_bps=c)["r_strategy"].mean()
             for c in (0.0, 10.0, 50.0)]
    assert means[0] >= means[1] >= means[2]


def test_borrow_only_bites_the_short_leg(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    long_flat = [st.run_rule(r["FUND1"], r["cash"], sig, direction=+1,
                             cost_bps=0.0, borrow_ann=b)["r_strategy"].mean()
                 for b in (0.0, 0.05)]
    short_paid = [st.run_rule(r["FUND1"], r["cash"], sig, direction=-1,
                              cost_bps=0.0, borrow_ann=b)["r_strategy"].mean()
                  for b in (0.0, 0.05)]
    assert long_flat[0] == pytest.approx(long_flat[1])   # no short leg, no borrow
    assert short_paid[0] > short_paid[1]                 # the short leg pays rent


def test_short_mirror_is_the_opposite_bet(planted):
    prices, _ = planted
    r = prices.pct_change()
    sig = st.trigger_signal(r["SPY"])
    lo = st.run_rule(r["FUND1"], r["cash"], sig, direction=+1, cost_bps=0.0)
    sh = st.run_rule(r["FUND1"], r["cash"], sig, direction=-1, cost_bps=0.0)
    assert lo["excess"].corr(sh["excess"]) < -0.9


def test_equal_weight_basket_handles_late_listings():
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"A": np.full(10, 0.01), "B": np.full(10, 0.03)}, index=idx)
    df.loc[idx[:5], "B"] = np.nan            # B "lists" halfway through
    b = st.equal_weight_basket(df, ["A", "B"])
    assert b.iloc[0] == pytest.approx(0.01)
    assert b.iloc[-1] == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# The study's spine — machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_catch_up_is_recovered(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices, truth)
    assert d["beta_mean"] == pytest.approx(truth["stale_beta_effective"], abs=0.05)
    assert d["t_beta_mean"] > 5


def test_planted_catch_up_pays_on_trigger_days(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices, truth)
    assert d["mean_on_bps"] > 5.0
    assert d["t_on_mean"] > 3.0


def test_null_produces_no_spurious_beta(null_panel):
    prices, truth = null_panel
    d = st.synthetic_detect(prices, truth)
    assert abs(d["beta_mean"]) < 0.05
    assert abs(d["t_beta_mean"]) < 2.0


def test_null_across_seeds_is_centered():
    betas = []
    for s in range(6):
        p, t = data.synthetic_panel(signal_strength=0.0, seed=917 + s)
        betas.append(st.synthetic_detect(p, t)["beta_mean"])
    betas = np.array(betas)
    assert abs(betas.mean()) < 0.02
    assert (np.abs(betas) >= 0.05).sum() == 0


def test_relative_regression_removes_the_shared_market_move(planted):
    """A pure beta-1 clone of SPY must show a zero *relative* loading by construction."""
    prices, _ = planted
    r = prices.pct_change()
    clone = r["SPY"].copy()
    reg = st.relative_regression(clone, r["SPY"])
    assert abs(reg["beta"]) < 1e-8


def test_domestic_control_runs(planted):
    prices, _ = planted
    reg = st.domestic_control(prices.pct_change()["SPY"])
    assert np.isfinite(reg["beta"]) and np.isfinite(reg["t_beta"])


# --------------------------------------------------------------------------- #
# Dropping the unit-beta ASSUMPTION
# --------------------------------------------------------------------------- #
def test_beta_hedged_recovers_the_planted_contemporaneous_beta(planted):
    """The fitted hedge ratio must be the panel's planted ``beta_contemp``."""
    prices, truth = planted
    r = prices.pct_change()
    hed = st.beta_hedged_regression(r["FUND1"], r["SPY"])
    assert hed["beta_contemp"] == pytest.approx(truth["beta_contemp"], abs=0.03)


def test_beta_hedged_still_recovers_the_planted_catch_up(planted):
    """Hedging the same-day move must not eat the LAGGED planted signal."""
    prices, truth = planted
    r = prices.pct_change()
    hed = st.beta_hedged_regression(r["FUND1"], r["SPY"])
    assert hed["beta"] == pytest.approx(truth["stale_beta_effective"], abs=0.05)
    assert hed["t_beta"] > 5


def test_beta_hedged_is_quiet_on_the_null(null_panel):
    prices, _ = null_panel
    r = prices.pct_change()
    hed = st.beta_hedged_regression(r["FUND1"], r["SPY"])
    assert abs(hed["beta"]) < 0.05
    assert abs(hed["t_beta"]) < 3


def test_unit_and_fitted_hedges_agree_only_when_beta_is_one():
    """The unit-beta shortcut is exact at beta=1 and biased away from it — by exactly
    ``(beta - 1) x b_domestic``, which is why the study reports both."""
    p1, _ = data.synthetic_panel(beta_contemp=1.0, signal_strength=0.0, seed=917)
    r1 = p1.pct_change()
    same = abs(st.relative_regression(r1["FUND1"], r1["SPY"])["beta"]
               - st.beta_hedged_regression(r1["FUND1"], r1["SPY"])["beta"])
    p2, _ = data.synthetic_panel(beta_contemp=0.5, signal_strength=0.0, seed=917)
    r2 = p2.pct_change()
    diff = abs(st.relative_regression(r2["FUND1"], r2["SPY"])["beta"]
               - st.beta_hedged_regression(r2["FUND1"], r2["SPY"])["beta"])
    assert same < 0.01
    assert diff > same


def test_confound_decomposition_is_additive(planted):
    """``b_raw = beta_c x b_domestic + b_hedged`` must hold as an identity, not a story."""
    prices, _ = planted
    r = prices.pct_change()
    d = st.confound_decomposition(r["FUND1"], r["SPY"])
    assert d["confound"] + d["beta_hedged"] == pytest.approx(d["beta_raw"], abs=1e-6)
    assert d["confound"] == pytest.approx(d["beta_contemp"] * d["beta_domestic"], abs=1e-9)


def test_per_fund_table_carries_both_hedges(planted):
    prices, truth = planted
    r = prices.pct_change()
    tbl = st.per_fund_regressions(r, truth["funds"], benchmark="SPY")
    for col in ("beta_raw", "beta_rel", "beta_hedged", "beta_contemp", "t_hedged"):
        assert col in tbl.columns
    assert np.isfinite(tbl["beta_hedged"]).all()


# --------------------------------------------------------------------------- #
# Robustness harness
# --------------------------------------------------------------------------- #
def test_compare_returns_every_arm(planted):
    prices, _ = planted
    r = prices.pct_change()
    cmp = st.compare(r["FUND1"], r["SPY"], r["cash"])
    for k in ("reg_raw", "reg_rel", "cond", "long", "short", "bh"):
        assert k in cmp
    assert 0.05 < cmp["trigger_frac"] < 0.15
    # the planted catch-up must make the LONG side the profitable one
    assert cmp["long"]["sharpe"] > cmp["short"]["sharpe"]


def test_era_cut_returns_both_halves(planted):
    prices, _ = planted
    r = prices.pct_change()
    eras = st.era_cut(r["FUND1"], r["SPY"], r["cash"], split="2010-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["beta_raw"]) and np.isfinite(e["t_rel"])
        # the fitted hedge must be refit WITHIN each era, not borrowed from the full sample
        assert np.isfinite(e["beta_hedged"]) and np.isfinite(e["beta_contemp"])


def test_cost_sweep_is_monotone(planted):
    prices, _ = planted
    r = prices.pct_change()
    rows = st.cost_sweep(r["FUND1"], r["SPY"], r["cash"], grid=(0.0, 10.0, 50.0))
    sharpes = [x["sharpe_long"] for x in rows]
    assert sharpes[0] > sharpes[1] > sharpes[2]


def test_borrow_sweep_is_monotone(planted):
    prices, _ = planted
    r = prices.pct_change()
    rows = st.borrow_sweep(r["FUND1"], r["SPY"], r["cash"], grid=(0.0, 0.02, 0.10))
    sharpes = [x["sharpe_short"] for x in rows]
    assert sharpes[0] > sharpes[1] > sharpes[2]


def test_threshold_sweep_shape(planted):
    prices, _ = planted
    r = prices.pct_change()
    rows = st.threshold_sweep(r["FUND1"], r["SPY"], r["cash"], grid=(0.80, 0.95))
    assert rows[0]["n_on"] > rows[1]["n_on"]
    assert all(np.isfinite(x["t_on"]) for x in rows)


def test_extra_lag_check_runs(planted):
    prices, _ = planted
    r = prices.pct_change()
    x = st.extra_lag_check(r["FUND1"], r["SPY"], r["cash"])
    assert np.isfinite(x["mean_on_bps"]) and x["n_on"] > 100


def test_bootstrap_ci_brackets_point(planted):
    prices, _ = planted
    r = prices.pct_change()
    e = (r["FUND1"] - r["cash"]).dropna()
    m = st.bootstrap_mean_ci(e, n_boot=400, seed=917)
    assert m["ci_low_bps"] <= m["mean_bps"] <= m["ci_high_bps"]
    s = st.bootstrap_sharpe_ci(e, n_boot=400, seed=917)
    assert s["ci_low"] <= s["sharpe"] <= s["ci_high"]


def test_summary_fields(planted):
    prices, _ = planted
    r = prices.pct_change()
    s = st.summary((r["FUND1"] - r["cash"]).dropna())
    for k in ("n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "tstat"):
        assert k in s and np.isfinite(s[k])
