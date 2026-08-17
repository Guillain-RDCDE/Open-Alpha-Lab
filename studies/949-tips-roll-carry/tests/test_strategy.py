"""Engine invariants and the study's spine — all offline and synthetic.

The spine: on a tape with a *planted* residual carry the duration-hedged sleeve recovers
it with a clearly significant HAC *t*; on the null it stays quiet across seeds. Around
that sit the honesty invariants — exactly one execution lag on the hedge beta, costs and
borrow that only ever subtract, and a ladder race that never mistakes duration for carry.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tips_roll import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Returns plumbing
# --------------------------------------------------------------------------- #
def test_excess_returns_are_net_of_cash(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    r = prices.pct_change().dropna()
    assert np.allclose(ex["tips"].to_numpy(), (r["tips"] - r["cash"]).to_numpy())
    # the cash column is kept as its own raw return, not zeroed out
    assert np.allclose(ex["cash"].to_numpy(), r["cash"].to_numpy())
    assert not ex.isna().any().any()


def test_summary_fields_and_signs(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    s = st.summary(ex["tips"])
    for k in ("n_days", "ann_return", "sharpe", "vol_ann", "cagr", "max_drawdown", "tstat"):
        assert k in s
    assert s["vol_ann"] > 0
    assert s["max_drawdown"] <= 0.0


def test_max_drawdown_matches_a_hand_case():
    r = pd.Series([0.10, -0.50, 0.10], index=pd.bdate_range("2020-01-01", periods=3))
    assert st.max_drawdown(r) == pytest.approx(-0.50)


# --------------------------------------------------------------------------- #
# The one execution lag — the hedge beta may not peek
# --------------------------------------------------------------------------- #
def test_rolling_beta_is_lagged_by_exactly_one_day(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    lagged = st.rolling_beta(ex["tips"], ex["nominal"], window=252)
    cov = ex["tips"].rolling(252).cov(ex["nominal"])
    unlagged = cov / ex["nominal"].rolling(252).var()
    aligned = lagged.dropna()
    assert np.allclose(aligned.to_numpy(), unlagged.shift(1).dropna().to_numpy())
    # window days of warmup plus the one-day shift
    assert lagged.iloc[:252].isna().all()


def test_no_lookahead_perturbing_the_future_leaves_the_past_alone(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    base = st.hedged_sleeve(ex["tips"], ex["nominal"])
    ex2 = ex.copy()
    ex2.iloc[2000:, ex2.columns.get_loc("tips")] *= 4.0
    pert = st.hedged_sleeve(ex2["tips"], ex2["nominal"])
    n = min(len(base), len(pert))
    # The sleeve drops the 252-day beta warmup, so sleeve row p is ex row p + 252;
    # 1700 is comfortably before the perturbation at ex row 2000.
    cut = 1700
    assert np.allclose(base["gross"].iloc[:cut].to_numpy(),
                       pert["gross"].iloc[:cut].to_numpy())
    assert n > cut


def test_static_alpha_recovers_the_planted_beta(planted):
    prices, truth = planted
    ex = st.excess_returns(prices, cash_col="cash")
    fit = st.static_alpha(ex["tips"], ex["nominal"])
    assert abs(fit["beta"] - truth["beta_true"]) < 0.10
    assert 0.0 < fit["r2"] < 1.0


# --------------------------------------------------------------------------- #
# Costs and borrow only ever subtract
# --------------------------------------------------------------------------- #
def test_costs_and_borrow_monotonically_reduce_the_net(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    means = []
    for c, b in [(0.0, 0.0), (2.0, 30.0), (10.0, 100.0)]:
        sl = st.hedged_sleeve(ex["tips"], ex["nominal"], cost_bps=c, borrow_bps_yr=b)
        means.append(sl["net"].mean())
    assert means[0] >= means[1] >= means[2]


def test_zero_frictions_leave_gross_equal_to_net(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    sl = st.hedged_sleeve(ex["tips"], ex["nominal"], cost_bps=0.0, borrow_bps_yr=0.0)
    assert np.allclose(sl["gross"].to_numpy(), sl["net"].to_numpy())


def test_borrow_charge_has_the_right_magnitude(planted):
    """A 100 bps/yr borrow on a ~beta=0.85 short leg must cost ~0.85%/yr, not more."""
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    free = st.sleeve_stats(st.hedged_sleeve(ex["tips"], ex["nominal"],
                                            cost_bps=0.0, borrow_bps_yr=0.0))
    paid = st.sleeve_stats(st.hedged_sleeve(ex["tips"], ex["nominal"],
                                            cost_bps=0.0, borrow_bps_yr=100.0))
    drag = free["net_ann"] - paid["net_ann"]
    assert 0.005 < drag < 0.013


# --------------------------------------------------------------------------- #
# The spine — the detector recovers a planted carry and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_carry_is_recovered(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices)
    assert d["gross_ann"] == pytest.approx(truth["carry_ann_planted"], abs=0.006)
    assert d["t_gross"] > 3.0


def test_planted_carry_recovered_by_the_static_fit_too(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices)
    assert d["static_alpha_ann"] == pytest.approx(truth["carry_ann_planted"], abs=0.006)
    assert d["static_t_alpha"] > 3.0


def test_null_is_quiet(null):
    prices, _ = null
    d = st.synthetic_detect(prices)
    assert abs(d["gross_ann"]) < 0.006
    assert abs(d["t_gross"]) < 2.0


def test_null_is_quiet_across_seeds():
    ts, alphas = [], []
    for s in range(8):
        prices, _ = data.synthetic_daily(signal_strength=0.0, seed=949 + s)
        d = st.synthetic_detect(prices)
        ts.append(d["t_gross"]); alphas.append(d["gross_ann"])
    ts = np.asarray(ts); alphas = np.asarray(alphas)
    assert abs(alphas.mean()) < 0.004          # centred on zero
    assert (np.abs(ts) >= 2.0).sum() <= 1      # ~5% false-positive rate, not more


def test_half_strength_carry_lands_in_between():
    full = st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=949)[0])
    half = st.synthetic_detect(data.synthetic_daily(signal_strength=0.5, seed=949)[0])
    nul = st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=949)[0])
    assert nul["gross_ann"] < half["gross_ann"] < full["gross_ann"]


# --------------------------------------------------------------------------- #
# The ladder race must not confuse duration with carry
# --------------------------------------------------------------------------- #
def test_ladder_race_shape(planted_panel):
    prices, truth = planted_panel
    ex = st.excess_returns(prices, cash_col="cash")
    legs = [f"tips_{i}" for i in range(truth["n_buckets"])]
    race = st.ladder_race(ex, legs)
    assert race["base"] == "tips_0"
    assert set(race["diff_vs_base"]) == set(legs[1:])
    for lg in legs:
        assert np.isfinite(race["legs"][lg]["sharpe"])


def test_flat_carry_term_structure_is_recovered_on_the_panel(planted_panel):
    """The panel plants the SAME carry in every bucket; the hedge must see it flat."""
    prices, truth = planted_panel
    d = st.synthetic_ladder_detect(prices, truth)
    carries = [d["carries"][f"bucket_{i}"]["gross_ann"] for i in range(truth["n_buckets"])]
    for c in carries:
        assert c == pytest.approx(truth["carry_ann_planted"], abs=0.008)
    assert max(carries) - min(carries) < 0.010


def test_null_panel_shows_no_carry_anywhere(null_panel):
    prices, truth = null_panel
    d = st.synthetic_ladder_detect(prices, truth)
    for i in range(truth["n_buckets"]):
        assert abs(d["carries"][f"bucket_{i}"]["gross_ann"]) < 0.008


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_newey_west_is_more_conservative_under_autocorrelation():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 0.01, 6000)
    x = np.zeros_like(e)
    for i in range(1, len(e)):
        x[i] = 0.0004 + 0.6 * x[i - 1] + e[i]
    assert abs(st.newey_west_t(x)) < abs(st.one_sample_t(x))


def test_welch_and_wilson_are_sane(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    assert np.isfinite(st.welch_t(ex["tips"].to_numpy(), ex["nominal"].to_numpy()))
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_intervals_bracket_their_points(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    sl = st.hedged_sleeve(ex["tips"], ex["nominal"])
    m = st.bootstrap_mean_ci(sl["gross"], n_boot=400, seed=949)
    s = st.bootstrap_sharpe_ci(sl["gross"], n_boot=400, seed=949)
    assert m["ci_low"] <= m["point"] <= m["ci_high"]
    assert s["ci_low"] <= s["sharpe"] <= s["ci_high"]


def test_bootstrap_flags_the_null_as_uncertain(null):
    prices, _ = null
    ex = st.excess_returns(prices, cash_col="cash")
    sl = st.hedged_sleeve(ex["tips"], ex["nominal"])
    m = st.bootstrap_mean_ci(sl["gross"], n_boot=400, seed=949)
    assert m["ci_low"] < 0.0 < m["ci_high"]


# --------------------------------------------------------------------------- #
# Robustness helpers
# --------------------------------------------------------------------------- #
def test_era_cut_returns_every_era(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    eras = [("first", "2013-01-01", "2018-12-31"), ("second", "2019-01-01", "2026-12-31")]
    out = st.era_cut(ex, "tips", "nominal", eras)
    assert set(out) == {"first", "second"}
    for e in out.values():
        assert e is not None and np.isfinite(e["net_ann"])


def test_era_cut_recovers_the_planted_carry_in_both_eras(planted):
    prices, truth = planted
    ex = st.excess_returns(prices, cash_col="cash")
    eras = [("first", "2013-01-01", "2018-12-31"), ("second", "2019-01-01", "2026-12-31")]
    out = st.era_cut(ex, "tips", "nominal", eras, cost_bps=0.0, borrow_bps_yr=0.0)
    for e in out.values():
        assert e["gross_ann"] == pytest.approx(truth["carry_ann_planted"], abs=0.012)


def test_drop_year_changes_nothing_much_when_the_carry_is_uniform(planted):
    prices, truth = planted
    ex = st.excess_returns(prices, cash_col="cash")
    kept = st.drop_year(ex, "tips", "nominal", 2018, cost_bps=0.0, borrow_bps_yr=0.0)
    assert kept["dropped_year"] == 2018
    assert kept["gross_ann"] == pytest.approx(truth["carry_ann_planted"], abs=0.008)


def test_cost_borrow_sweep_is_monotone(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    sw = st.cost_borrow_sweep(ex, "tips", "nominal")
    c = [row["net_ann"] for row in sw["cost_sweep"]]
    b = [row["net_ann"] for row in sw["borrow_sweep"]]
    assert c == sorted(c, reverse=True)
    assert b == sorted(b, reverse=True)


def test_window_sweep_beta_is_stable(planted):
    prices, truth = planted
    ex = st.excess_returns(prices, cash_col="cash")
    for row in st.window_sweep(ex, "tips", "nominal"):
        assert abs(row["mean_beta"] - truth["beta_true"]) < 0.15


def test_calendar_year_table_shape(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, cash_col="cash")
    tbl = st.calendar_year_table(st.hedged_sleeve(ex["tips"], ex["nominal"]))
    assert {"gross_pct", "net_pct", "n_days"}.issubset(tbl.columns)
    assert len(tbl) > 5
    assert (tbl["net_pct"] <= tbl["gross_pct"] + 1e-9).all()


def test_sleeve_table_covers_every_pair(planted_panel):
    prices, truth = planted_panel
    ex = st.excess_returns(prices, cash_col="cash")
    pairs = [(f"tips_{i}", f"nom_{i}") for i in range(truth["n_buckets"])]
    tbl = st.sleeve_table(ex, pairs)
    assert set(tbl) == set(pairs)
    for v in tbl.values():
        assert np.isfinite(v["net_ann"]) and v["n_days"] > 100
