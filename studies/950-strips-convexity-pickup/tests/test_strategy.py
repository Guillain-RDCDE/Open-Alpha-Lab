"""Hedge logic, race invariants, inference primitives and the study's spine — all offline.

The spine: on a world with a *planted* convexity pickup the asymmetry regression must
recover a positive quadratic coefficient and a negative intercept (convexity is real and
is paid for); on the null — where a duration-matched mix is convexity-matched too — it
must stay quiet. Around that: the duration match must actually neutralise duration, the
hedge ratio must use only past data (exactly one execution lag), costs must fall on the
mix and only on the mix, and the financing spread must move the spread in the direction
that flatters the zero leg.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_convexity import data, strategy as st  # noqa: E402


def _race(panel, **kw):
    return st.run_race(panel["zero"], panel["coupon"], panel["cash"], panel["yield_pp"], **kw)


# --------------------------------------------------------------------------- #
# The rate factor and the duration match
# --------------------------------------------------------------------------- #
def test_rate_factor_is_a_diff_in_decimals():
    y = pd.Series([4.00, 4.10, 3.95], index=pd.bdate_range("2020-01-01", periods=3))
    dy = st.rate_factor(y)
    assert np.isnan(dy.iloc[0])
    assert dy.iloc[1] == pytest.approx(0.0010)
    assert dy.iloc[2] == pytest.approx(-0.0015)


def test_rolling_rate_beta_recovers_minus_duration():
    """A leg built as r = -D*dy must show a rolling slope of -D."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=600)
    dy = pd.Series(rng.normal(0, 5e-4, 600), index=idx)
    r = -18.0 * dy
    beta = st.rolling_rate_beta(r, dy, window=252)
    assert beta.iloc[:251].isna().all()          # full window required, no partials
    assert beta.dropna().iloc[0] == pytest.approx(-18.0, abs=1e-6)


def test_hedge_ratio_is_constant_within_a_month_and_lagged():
    idx = pd.bdate_range("2020-01-01", periods=300)
    bz = pd.Series(np.linspace(-24.0, -20.0, 300), index=idx)
    bc = pd.Series(np.full(300, -16.0), index=idx)
    L = st.monthly_hedge_ratio(bz, bc)
    per = L.index.to_period("M")
    for p in pd.unique(per)[1:]:
        vals = L[per == p].dropna().unique()
        assert len(vals) <= 1                     # one weight per month
    # The weight traded in month m+1 is the ratio observed on the last day of month m.
    ratio = bz / bc
    months = list(pd.unique(per))
    prev_last = idx[per == months[1]][-1]
    first_next = idx[per == months[2]][0]
    assert L.loc[first_next] == pytest.approx(ratio.loc[prev_last])
    assert L[per == months[0]].isna().all()       # nothing to trade in the first month


def test_hedge_ratio_uses_no_future_data(planted):
    """Perturbing the tail of the tape must not change any earlier hedge weight."""
    panel, _ = planted
    race = _race(panel)
    bumped = panel.copy()
    bumped.iloc[3000:, bumped.columns.get_loc("zero")] *= 1.5
    race2 = _race(bumped)
    cut = race.index[2500]
    a = race.loc[:cut, "L"]
    b = race2.loc[:cut, "L"]
    assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy(), equal_nan=True)


def test_hedge_ratio_rejects_absurd_estimates():
    idx = pd.bdate_range("2020-01-01", periods=100)
    bz = pd.Series(np.full(100, -24.0), index=idx)
    bc = pd.Series(np.full(100, -0.01), index=idx)   # ratio 2400 -> unusable
    assert st.monthly_hedge_ratio(bz, bc).dropna().empty


# --------------------------------------------------------------------------- #
# The race
# --------------------------------------------------------------------------- #
def test_race_columns_and_no_nans(planted):
    panel, _ = planted
    race = _race(panel)
    assert {"e_zero", "e_mix", "diff", "L", "dy", "turnover"}.issubset(race.columns)
    assert not race.isna().any().any()
    assert len(race) > 2000


def test_duration_match_equalises_volatility(planted):
    """The whole premise: after matching, the two arms carry the same rate risk."""
    panel, _ = planted
    race = _race(panel)
    assert race["e_zero"].std() / race["e_mix"].std() == pytest.approx(1.0, abs=0.05)


def test_hedge_ratio_recovers_the_planted_duration_ratio(planted):
    panel, truth = planted
    race = _race(panel)
    assert race["L"].mean() == pytest.approx(truth["duration_ratio"], abs=0.05)


def test_costs_fall_on_the_mix_and_only_on_the_mix(planted):
    panel, _ = planted
    lo = _race(panel, cost_bps=0.0)
    hi = _race(panel, cost_bps=25.0)
    # The zero arm is buy-and-hold: identical under any cost assumption.
    assert np.allclose(lo["e_zero"].to_numpy(), hi["e_zero"].to_numpy())
    # The mix pays the turnover, so it earns less and the spread widens in its favour.
    assert hi["e_mix"].mean() < lo["e_mix"].mean()
    assert hi["diff"].mean() > lo["diff"].mean()


def test_financing_spread_moves_the_spread_towards_the_zero(planted):
    panel, _ = planted
    lo = _race(panel, finance_bps=0.0)
    hi = _race(panel, finance_bps=100.0)
    assert hi["diff"].mean() > lo["diff"].mean()
    # ... and the effect is the levered fraction times the spread, to the basis point.
    lev = (lo["L"] - 1.0).mean()
    expected = lev * (100.0 * 1e-4) / st.TRADING_DAYS
    assert (hi["diff"].mean() - lo["diff"].mean()) == pytest.approx(expected, rel=0.05)


def test_monthly_aggregation_shapes(planted):
    panel, _ = planted
    race = _race(panel)
    m = st.to_monthly(race, panel["yield_pp"])
    assert {"e_zero", "e_mix", "diff", "dy", "rv", "L"}.issubset(m.columns)
    assert (m["rv"] > 0).all()
    # The two quadratic regressors are different objects but of the same magnitude: with
    # near-serially-independent daily moves, E[(sum dy)^2] = E[sum dy^2].
    assert 0.5 < m["rv"].mean() / (m["dy"] ** 2).mean() < 2.0
    assert 100 < len(m) < 300


def test_summary_reports_finite_stats(planted):
    panel, _ = planted
    race = _race(panel)
    s = st.summary(race["diff"])
    for k in ("mean_ann", "vol_ann", "sharpe", "cagr", "max_drawdown", "tstat"):
        assert np.isfinite(s[k])
    assert s["max_drawdown"] <= 0.0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_planted_coefficients():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.normal(0, 1, n)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, n)
    fit = st.hac_ols(y, np.column_stack([np.ones(n), x]), lags=4)
    assert fit["beta"][0] == pytest.approx(0.5, abs=0.05)
    assert fit["beta"][1] == pytest.approx(2.0, abs=0.05)
    assert fit["t"][1] > 20 and 0.9 < fit["r2"] < 1.0


def test_welch_and_one_sample_t_are_finite(planted):
    panel, _ = planted
    race = _race(panel)
    assert np.isfinite(st.one_sample_t(race["diff"].to_numpy()))
    assert np.isfinite(st.welch_t(race["e_zero"].to_numpy(), race["e_mix"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(44, 100)
    assert lo < 0.44 < hi


def test_block_bootstrap_brackets_the_point(planted):
    panel, _ = planted
    m = st.to_monthly(_race(panel), panel["yield_pp"])
    ci = st.block_bootstrap_ci(m["diff"], n_boot=400, seed=950)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    shp = st.block_bootstrap_ci(m["diff"], stat="sharpe", n_boot=400, seed=950)
    assert shp["ci_low"] <= shp["point"] <= shp["ci_high"]


def test_bootstrap_b2_brackets_the_point(planted):
    panel, _ = planted
    m = st.to_monthly(_race(panel), panel["yield_pp"])
    b = st.bootstrap_b2_ci(m, n_boot=400, seed=950, regressor="rv")
    reg = st.convexity_regression(m, regressor="rv")
    assert b["point"] == pytest.approx(reg["b2"], rel=1e-6)
    assert b["ci_low"] <= b["point"] <= b["ci_high"]


# --------------------------------------------------------------------------- #
# Descriptive helpers
# --------------------------------------------------------------------------- #
def test_move_buckets_are_ordered_by_move_size(planted):
    panel, _ = planted
    m = st.to_monthly(_race(panel), panel["yield_pp"])
    tbl = st.move_buckets(m)
    assert list(tbl.index) == ["quiet", "middling", "large"]
    assert tbl["mean_absdy_bp"].is_monotonic_increasing
    assert tbl["n"].sum() == len(m)


def test_linear_hedge_removes_the_linear_exposure(planted):
    panel, _ = planted
    m = st.to_monthly(_race(panel), panel["yield_pp"])
    hedged = st.linearly_hedged_spread(m)
    X = np.column_stack([np.ones(len(m)), m["dy"].to_numpy()])
    slope = (np.linalg.pinv(X.T @ X) @ X.T @ hedged.to_numpy())[1]
    assert abs(slope) < 1e-9


def test_breakeven_move_is_consistent_with_the_fit(planted):
    panel, _ = planted
    m = st.to_monthly(_race(panel), panel["yield_pp"])
    reg = st.convexity_regression(m, regressor="rv")
    if np.isfinite(reg["breakeven_move_bp"]):
        dy = reg["breakeven_move_bp"] * 1e-4
        assert reg["a_bp_mo"] * 1e-4 + reg["b2"] * dy ** 2 == pytest.approx(0.0, abs=1e-9)


def test_sweeps_cover_their_grids(planted):
    panel, _ = planted
    z, c, k, y = panel["zero"], panel["coupon"], panel["cash"], panel["yield_pp"]
    costs = st.sweep_costs(z, c, k, y, grid=(0.0, 10.0))
    fins = st.sweep_finance(z, c, k, y, grid=(0.0, 50.0))
    wins = st.sweep_window(z, c, k, y, grid=(126, 252))
    assert [r["cost_bps"] for r in costs] == [0.0, 10.0]
    assert [r["finance_bps"] for r in fins] == [0.0, 50.0]
    assert [r["window"] for r in wins] == [126, 252]
    assert fins[1]["mean_diff_bp_mo"] > fins[0]["mean_diff_bp_mo"]


def test_cut_grid_covers_every_combination(planted):
    """The grid must be a census: every fund x era x spec, no cut quietly dropped."""
    panel, _ = planted
    race = _race(panel)
    grid = st.cut_grid({"zeroA": race, "zeroB": race}, panel["yield_pp"], split="2017-01-01")
    assert len(grid) == 2 * 3 * 2                      # 2 funds x 3 eras x 2 specs
    assert set(grid["era"]) == {"full", "early", "late"}
    assert set(grid["spec"]) == {"dy2", "rv"}
    assert set(grid["fund"]) == {"zeroA", "zeroB"}
    assert np.isfinite(grid[["a_bp_mo", "a_t", "b1_t", "b2", "b2_t"]].to_numpy()).all()


def test_grid_census_reports_the_true_extreme(planted):
    """The census must quote the largest |t| in the WHOLE grid, not the best cut on show."""
    panel, _ = planted
    race = _race(panel)
    grid = st.cut_grid({"zero": race}, panel["yield_pp"], split="2017-01-01")
    cen = st.grid_census(grid)
    assert cen["n_cuts"] == len(grid)
    assert cen["max_abs_b2_t"] == pytest.approx(grid["b2_t"].abs().max())
    assert cen["n_b2_t_ge_2"] == int((grid["b2_t"].abs() >= 2.0).sum())
    assert cen["b2_positive"] + cen["sign_flips"] >= cen["n_cuts"]
    # On the planted world every cut should agree with the plant.
    assert cen["b2_positive"] == cen["n_cuts"]
    assert cen["a_negative"] == cen["n_cuts"]


def test_grid_census_counts_a_sign_flip(planted, null_world):
    """A grid holding a genuinely flipped cut must be reported as flipped, not rounded away."""
    panel, _ = planted
    nul, _ = null_world
    grid = st.cut_grid({"planted": _race(panel)}, panel["yield_pp"], split="2017-01-01")
    gnul = st.cut_grid({"null": _race(nul)}, nul["yield_pp"], split="2017-01-01")
    both = pd.concat([grid, gnul], ignore_index=True)
    cen = st.grid_census(both)
    assert cen["n_cuts"] == len(both)
    assert cen["sign_flips"] == int(((both["b2"] <= 0) | (both["a_bp_mo"] >= 0)).sum())
    # The null half is not obliged to agree with the plant, so the census must not claim
    # uniformity across the pooled grid.
    assert cen["b2_positive"] <= cen["n_cuts"]


def test_era_cut_returns_both_halves(planted):
    panel, _ = planted
    race = _race(panel)
    eras = st.era_cut(race, panel["yield_pp"], split="2017-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["b2"]) and np.isfinite(e["mean_diff_bp_mo"])


# --------------------------------------------------------------------------- #
# The spine — the detector fires on a planted pickup and is silent on the null
# --------------------------------------------------------------------------- #
def test_planted_pickup_is_recovered(planted):
    panel, truth = planted
    d = st.synthetic_detect(panel)
    # A positive quadratic coefficient with a convincing t: convexity is really there ...
    assert d["b2"] > 0 and d["b2_t"] > 3.0
    # ... and it is paid for: a negative intercept of roughly the planted carry give-up.
    assert d["a_bp_mo"] < 0
    assert d["a_bp_mo"] == pytest.approx(-truth["carry_giveup_bp_mo"], abs=8.0)
    # The linear term must be neutralised by the duration match, not left to soak it up.
    assert abs(d["b1_t"]) < 3.0


def test_planted_pickup_size_matches_the_plant(planted):
    """b2 should be about 0.5*(C_long - L*C_short), the second-order term of the spread."""
    panel, truth = planted
    d = st.synthetic_detect(panel)
    expected = 0.5 * (truth["convexity_long"]
                      - truth["duration_ratio"] * truth["convexity_short"])
    assert d["b2"] == pytest.approx(expected, rel=0.5)


def test_null_shows_no_convexity(null_world):
    panel, truth = null_world
    d = st.synthetic_detect(panel)
    assert truth["convexity_per_dur_long"] == pytest.approx(truth["convexity_per_dur_short"])
    assert abs(d["b2_t"]) < 3.0
    assert abs(d["a_t"]) < 3.0


def test_null_is_centred_across_seeds():
    ts = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=950 + s)[0])["b2_t"]
        for s in range(8)
    ])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 2.0).sum() <= 2


def test_planted_beats_null_across_seeds():
    def b2t(ss, s):
        return st.synthetic_detect(data.synthetic_panel(signal_strength=ss, seed=950 + s)[0])["b2_t"]
    planted_ts = np.array([b2t(1.0, s) for s in range(4)])
    null_ts = np.array([b2t(0.0, s) for s in range(4)])
    assert planted_ts.min() > 2.0
    assert planted_ts.mean() > null_ts.mean() + 3.0
