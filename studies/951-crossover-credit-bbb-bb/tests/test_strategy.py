"""Estimator invariants, execution-lag discipline, and the study's spine — all offline.

The spine: on a ladder where the crossover rung is genuinely paid, the two-factor
head-to-head estimator recovers the planted premium with a clear HAC *t*; on the null
ladder — where the raw excess-return gaps are still large, because the rungs load
differently on drifting factors — the adjusted alphas are centred on zero and do not fire.
Costs and borrow monotonically reduce the spread; the long/short expression trades on the
month-end signal one day later; the robustness tools return coherent shapes.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossover_credit import data, strategy as st  # noqa: E402

FACT = ("dur", "eq")
CASH = "cash"


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_default_hac_lags_follows_the_rule_of_thumb():
    assert st.default_hac_lags(3574) == int(np.floor(4.0 * (3574 / 100.0) ** (2.0 / 9.0)))
    assert st.default_hac_lags(10) >= 1


def test_ols_hac_recovers_known_coefficients():
    rng = np.random.default_rng(951)
    n = 4000
    f1 = rng.normal(0, 0.01, n)
    f2 = rng.normal(0, 0.011, n)
    y = 0.0002 + 0.4 * f1 + 0.7 * f2 + rng.normal(0, 0.002, n)
    res = st.ols_hac(y, np.column_stack([f1, f2]))
    assert res["beta"][0] == pytest.approx(0.0002, abs=1e-4)
    assert res["beta"][1] == pytest.approx(0.4, abs=0.05)
    assert res["beta"][2] == pytest.approx(0.7, abs=0.05)
    assert res["tstat"][1] > 5 and res["tstat"][2] > 5
    assert 0.0 <= res["r2"] <= 1.0


def test_ols_hac_intercept_only_matches_the_hac_mean_t():
    rng = np.random.default_rng(7)
    y = 0.0003 + rng.normal(0, 0.004, 3000)
    res = st.ols_hac(y, np.empty((3000, 0)))
    assert res["tstat"][0] == pytest.approx(st.newey_west_t(y), rel=1e-6)


def test_one_sample_t_is_sane():
    rng = np.random.default_rng(3)
    assert st.one_sample_t(rng.normal(0.002, 0.01, 2000)) > 3
    assert abs(st.one_sample_t(rng.normal(0.0, 0.01, 2000))) < 3


# --------------------------------------------------------------------------- #
# Ladder / pair mechanics
# --------------------------------------------------------------------------- #
def test_ladder_table_recovers_the_planted_loadings(paid_rung):
    prices, truth = paid_rung
    tbl = st.ladder_table(prices, rungs=("ig", "crossover", "hy"), factors=FACT, cash=CASH)
    assert list(tbl.index) == ["ig", "crossover", "hy"]
    for rung in ("ig", "crossover", "hy"):
        b_dur, b_eq = truth["loadings"][rung]
        assert tbl.loc[rung, "beta_dur"] == pytest.approx(b_dur, abs=0.05)
        assert tbl.loc[rung, "beta_eq"] == pytest.approx(b_eq, abs=0.05)
    assert (tbl["max_drawdown_abs"] <= 0).all()


def test_pair_alpha_difference_is_the_plain_difference(paid_rung):
    prices, _ = paid_rung
    res = st.pair_alpha(prices, "crossover", "hy", factors=FACT, cash=CASH)
    e = st.excess_returns(prices, cash=CASH)
    assert np.allclose(res["diff"].to_numpy(), (e["crossover"] - e["hy"]).to_numpy())
    assert res["raw_diff_ann"] == pytest.approx(float(res["diff"].mean() * st.TRADING_DAYS))


def test_pair_alpha_is_antisymmetric(paid_rung):
    prices, _ = paid_rung
    a = st.pair_alpha(prices, "crossover", "hy", factors=FACT, cash=CASH)
    b = st.pair_alpha(prices, "hy", "crossover", factors=FACT, cash=CASH)
    assert a["alpha_ann"] == pytest.approx(-b["alpha_ann"], abs=1e-12)


def test_summary_and_max_drawdown_are_finite(paid_rung):
    prices, _ = paid_rung
    e = st.excess_returns(prices, cash=CASH)
    s = st.summary(e["crossover"])
    for k in ("ann_return", "sharpe", "vol_ann", "tstat"):
        assert np.isfinite(s[k])
    assert st.max_drawdown(e["crossover"]) <= 0.0


# --------------------------------------------------------------------------- #
# The study's spine — planted premium recovered, null stays quiet
# --------------------------------------------------------------------------- #
def test_planted_premium_is_recovered(paid_rung):
    prices, truth = paid_rung
    d = st.synthetic_detect(prices)
    planted = truth["alpha_planted"]
    assert d["alpha_ann"] == pytest.approx(planted, abs=0.6 * planted)
    assert d["t_alpha"] > 2.0


def test_planted_premium_shows_against_the_ig_rung_too(paid_rung):
    prices, truth = paid_rung
    d = st.synthetic_detect(prices)
    assert d["alpha_vs_ig"] > 0.5 * truth["alpha_planted"]
    assert d["t_vs_ig"] > 2.0


def test_null_ladder_produces_no_adjusted_alpha(flat_ladder):
    """On the null the raw gaps can be large; the *adjusted* alpha must not fire."""
    prices, _ = flat_ladder
    e = st.excess_returns(prices, cash=CASH)
    raw_gap = float((e["crossover"] - e["hy"]).mean() * st.TRADING_DAYS)
    d = st.synthetic_detect(prices)
    assert abs(raw_gap) > 0.0            # the rungs really do differ before adjustment
    assert abs(d["t_alpha"]) < 2.0
    assert abs(d["t_vs_ig"]) < 2.0


def test_null_across_seeds_is_centered_and_quiet():
    alphas, ts = [], []
    for s in range(8):
        px, _ = data.synthetic_panel(signal_strength=0.0, seed=951 + s)
        d = st.synthetic_detect(px)
        alphas.append(d["alpha_ann"])
        ts.append(d["t_alpha"])
    alphas, ts = np.array(alphas), np.array(ts)
    assert abs(alphas.mean()) < 0.01           # < 1 %/yr on average
    assert (np.abs(ts) >= 2.0).sum() <= 1      # a 5% test may fire once in eight


def test_half_strength_lands_between_null_and_full():
    full = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=951)[0])
    half = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=951)[0])
    null = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=951)[0])
    assert null["alpha_ann"] < half["alpha_ann"] < full["alpha_ann"]


# --------------------------------------------------------------------------- #
# Bootstrap, eras, jackknife, bandwidth
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_point(paid_rung):
    prices, _ = paid_rung
    res = st.pair_alpha(prices, "crossover", "hy", factors=FACT, cash=CASH)
    ci = st.bootstrap_alpha_ci(res["diff"], res["factors"], n_boot=400, seed=951)
    assert ci["ci_low"] <= ci["alpha_ann"] <= ci["ci_high"]
    assert ci["alpha_ann"] == pytest.approx(res["alpha_ann"], rel=1e-6)
    assert 0.0 <= ci["frac_negative"] <= 1.0


def test_era_cut_returns_both_halves(paid_rung):
    prices, _ = paid_rung
    split = str(prices.index[len(prices) // 2].date())
    eras = st.era_cut(prices, "crossover", "hy", split=split, factors=FACT, cash=CASH)
    assert eras["early"] is not None and eras["late"] is not None
    assert eras["early"]["n"] + eras["late"]["n"] <= len(prices)
    for r in eras.values():
        assert np.isfinite(r["alpha_ann"]) and np.isfinite(r["t_alpha"])


def test_year_jackknife_drops_one_year_each_time(paid_rung):
    prices, _ = paid_rung
    jack = st.year_jackknife(prices, "crossover", "hy", factors=FACT, cash=CASH)
    years = sorted(set(prices.index.year))
    assert list(jack.index) == years
    assert (jack["n"] < len(prices)).all()
    # A broad, evenly-spread planted premium must survive every single-year deletion.
    assert (jack["t_alpha"] > 2.0).all()


def test_drop_years_matches_a_manual_mask(paid_rung):
    prices, _ = paid_rung
    years = sorted(set(prices.index.year))[:2]
    res = st.drop_years(prices, "crossover", "hy", years, factors=FACT, cash=CASH)
    e = st.excess_returns(prices, cash=CASH)
    keep = ~pd.Index(e.index.year).isin(years)
    assert res["n"] == int(keep.sum())
    assert res["dropped"] == list(years)


def test_calendar_year_table_covers_every_year(paid_rung):
    prices, _ = paid_rung
    cal = st.calendar_year_table(prices, "crossover", "hy", cash=CASH)
    assert list(cal.index) == sorted(set(prices.index.year))
    assert cal["diff_pct"].notna().all()


def test_lag_sensitivity_leaves_the_point_estimate_alone(paid_rung):
    """The HAC bandwidth changes the standard error, never the alpha itself."""
    prices, _ = paid_rung
    res = st.pair_alpha(prices, "crossover", "hy", factors=FACT, cash=CASH)
    tbl = st.lag_sensitivity(res["diff"], res["factors"], lag_grid=(5, 10, 21, 42))
    assert list(tbl.index) == [5, 10, 21, 42]
    assert tbl["alpha_ann"].std() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# The tradable expressions — one execution lag, costs, borrow
# --------------------------------------------------------------------------- #
def test_long_only_swap_reports_both_legs(paid_rung):
    prices, _ = paid_rung
    swap = st.long_only_swap(prices, "crossover", "hy", cost_bps=5.0, cash=CASH)
    assert swap["sharpe_gain"] == pytest.approx(swap["sharpe_long"] - swap["sharpe_short"])
    assert swap["dd_long_abs"] <= 0 and swap["dd_short_abs"] <= 0


def test_long_only_swap_cost_is_amortised_not_repeated(paid_rung):
    """A one-off switch must cost the same in total however long the sample is held."""
    prices, _ = paid_rung
    cheap = st.long_only_swap(prices, "crossover", "hy", cost_bps=0.0, cash=CASH)
    dear = st.long_only_swap(prices, "crossover", "hy", cost_bps=50.0, cash=CASH)
    drag = cheap["excess_ann_long"] - dear["excess_ann_long"]
    assert drag > 0
    assert drag < 1e-3           # < 10 bps/yr for a 50 bps one-off over 14 years


def test_long_short_spread_borrow_and_cost_are_monotone(paid_rung):
    prices, _ = paid_rung
    means = [st.long_short_spread(prices, "crossover", "hy", borrow_bps=b, cost_bps=5.0,
                                  factors=FACT, cash=CASH)["mean_ann"]
             for b in (0.0, 50.0, 200.0)]
    assert means[0] > means[1] > means[2]
    costs = [st.long_short_spread(prices, "crossover", "hy", borrow_bps=0.0, cost_bps=c,
                                  factors=FACT, cash=CASH)["mean_ann"]
             for c in (0.0, 5.0, 50.0)]
    assert costs[0] >= costs[1] >= costs[2]


def test_long_short_spread_charges_borrow_at_the_stated_rate(paid_rung):
    """200 bps of borrow on a dollar-neutral short leg must cost ~200 bps/yr, no more."""
    prices, _ = paid_rung
    free = st.long_short_spread(prices, "crossover", "hy", borrow_bps=0.0, cost_bps=0.0,
                                factors=FACT, cash=CASH)
    paid = st.long_short_spread(prices, "crossover", "hy", borrow_bps=200.0, cost_bps=0.0,
                                factors=FACT, cash=CASH)
    assert (free["mean_ann"] - paid["mean_ann"]) == pytest.approx(0.02, abs=0.004)


def test_long_short_spread_uses_the_month_end_signal_one_day_later(paid_rung):
    """Perturbing the tail of the tape must not change the spread before the perturbation."""
    prices, _ = paid_rung
    cut = 2000
    perturbed = prices.copy()
    perturbed.iloc[cut:, :] = perturbed.iloc[cut:, :] * 3.0
    a = st.long_short_spread(prices, "crossover", "hy", factors=FACT, cash=CASH)["net"]
    b = st.long_short_spread(perturbed, "crossover", "hy", factors=FACT, cash=CASH)["net"]
    common = a.index[a.index < prices.index[cut]]
    assert len(common) > 500
    assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy())


def test_borrow_sweep_is_ordered(paid_rung):
    prices, _ = paid_rung
    sweep = st.borrow_sweep(prices, "crossover", "hy", borrow_grid=(0.0, 50.0, 100.0),
                            factors=FACT, cash=CASH)
    assert list(sweep.index) == [0.0, 50.0, 100.0]
    assert sweep["alpha_ann"].iloc[0] > sweep["alpha_ann"].iloc[-1]
    assert (sweep["turnover_per_year"] >= 0).all()
