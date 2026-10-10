"""Strategy tests for Study 1026 — Gibson vs Fisher, synthetic-first, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gibson import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch / statsmodels tapes not installed")


# --------------------------------------------------------------------------- #
# The regression engine
# --------------------------------------------------------------------------- #
def test_hac_ols_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, 2000)
    d = st.hac_ols(y, x)
    assert d["coef"][0] == pytest.approx(0.5, abs=0.05)
    assert d["coef"][1] == pytest.approx(2.0, abs=0.05)
    assert d["t"][1] > 50


def test_hac_widens_standard_errors_under_autocorrelation():
    rng = np.random.default_rng(2)
    n = 2000
    e = np.zeros(n)
    x = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
        x[t] = 0.8 * x[t - 1] + rng.normal()
    y = 0.3 * x + e
    naive = st.hac_ols(y, x, lags=0)
    hac = st.hac_ols(y, x, lags=24)
    assert hac["se"][1] > 1.3 * naive["se"][1]


def test_hac_ols_declines_tiny_samples_and_hac_mean_handles_them():
    assert st.hac_ols(np.arange(5.0), np.arange(5.0)) == {}
    assert np.isnan(st.hac_mean(np.arange(5.0))["t"])


# --------------------------------------------------------------------------- #
# The spurious-regression trap, demonstrated
# --------------------------------------------------------------------------- #
def test_independent_random_walks_fool_levels_but_not_the_gates():
    """Granger-Newbold, in miniature: big naive t in levels; differences and EG see nothing."""
    rng = np.random.default_rng(1974)
    naive_big, eg_rejects, diff_rejects = 0, 0, 0
    reps = 20
    for _ in range(reps):
        a = pd.Series(np.cumsum(rng.normal(size=400)))
        b = pd.Series(np.cumsum(rng.normal(size=400)))
        eg = st.engle_granger(a, b)
        naive_big += abs(eg["naive_t"]) > 2
        eg_rejects += eg["eg_p"] < 0.05
        d = st.hac_ols(a.diff().to_numpy(), b.diff().to_numpy(), lags=6)
        diff_rejects += abs(d["t"][1]) > 2
    assert naive_big >= 0.6 * reps          # the trap springs most of the time
    assert eg_rejects <= 0.2 * reps         # cointegration does not
    assert diff_rejects <= 0.2 * reps       # nor do differences


def test_unit_root_table_labels_walks_and_noise():
    rng = np.random.default_rng(3)
    walk = pd.Series(np.cumsum(rng.normal(size=500)))
    noise = pd.Series(rng.normal(size=500))
    ur = st.unit_root_table({"walk": walk, "noise": noise})
    assert bool(ur.loc["walk", "i1_like"]) is True
    assert bool(ur.loc["noise", "i1_like"]) is False


def test_engle_granger_finds_a_planted_cointegration():
    rng = np.random.default_rng(4)
    x = pd.Series(np.cumsum(rng.normal(size=500)))
    y = 1.0 + 0.7 * x + pd.Series(rng.normal(0, 0.5, 500))
    eg = st.engle_granger(y, x)
    assert eg["eg_p"] < 0.01
    assert eg["slope"] == pytest.approx(0.7, abs=0.05)


# --------------------------------------------------------------------------- #
# The detector: fires in its own world, quiet in the other
# --------------------------------------------------------------------------- #
def test_gibson_world_passes_every_gibson_leg(gibson_legs):
    assert gibson_legs["gibson"] == {"L1": True, "L2": True, "L3": True}
    assert gibson_legs["gibson_strong"]


def test_fisher_world_passes_no_gibson_leg(fisher_legs):
    assert fisher_legs["gibson"] == {"L1": False, "L2": False, "L3": False}
    assert fisher_legs["fisher_strong"]


def test_horse_race_assigns_credit_to_the_right_variable(gibson_legs, fisher_legs):
    g, f = gibson_legs["horse_race"], fisher_legs["horse_race"]
    assert g["t_gap"] > 2 and abs(g["t_pi"]) < 2
    assert f["t_pi"] > 2 and abs(f["t_gap"]) < 2


def test_cointegration_points_at_the_planted_law(gibson_legs, fisher_legs):
    assert gibson_legs["eg_gap"]["eg_p"] < 0.05 < gibson_legs["eg_pi"]["eg_p"]
    assert fisher_legs["eg_pi"]["eg_p"] < 0.05 < fisher_legs["eg_gap"]["eg_p"]


def test_signal_stamp_on_the_two_worlds(gibson_legs, fisher_legs):
    assert st.signal_stamp(gibson_legs["gibson"], gibson_legs["gibson"]) == "Real"
    assert st.signal_stamp(fisher_legs["gibson"], fisher_legs["gibson"]) == "None"


# --------------------------------------------------------------------------- #
# Out of sample — no look-ahead
# --------------------------------------------------------------------------- #
def test_oos_forecast_never_uses_the_future(gibson_world):
    df = gibson_world[0].iloc[:360].copy()
    f1 = st.oos_forecast(df, "aaa", ["gap_exp"], h=12, min_train=60)
    df2 = df.copy()
    df2.iloc[300:, df2.columns.get_loc("aaa")] += 25.0      # rewrite the future
    f2 = st.oos_forecast(df2, "aaa", ["gap_exp"], h=12, min_train=60)
    cut = df.index[300 - 12]                                  # last origin blind to the change
    a = f1.loc[:cut, ["f_mean", "f_base", "f_full"]]
    b = f2.loc[:cut, ["f_mean", "f_base", "f_full"]]
    pd.testing.assert_frame_equal(a, b)


def test_publication_lag_shifts_only_price_columns():
    df = pd.DataFrame({"aaa": [1.0, 2, 3], "gap_exp": [10.0, 20, 30], "pi": [5.0, 6, 7]})
    out = st.add_publication_lag(df, cols=("gap_exp", "pi"), lag=1)
    assert out["aaa"].tolist() == [1.0, 2, 3]
    assert np.isnan(out["gap_exp"].iloc[0]) and out["gap_exp"].iloc[1] == 10.0


def test_clark_west_rewards_information_and_ignores_noise():
    rng = np.random.default_rng(5)
    n = 600
    signal = rng.normal(size=n)
    actual = 0.6 * signal + rng.normal(size=n)
    small = np.zeros(n)
    good = 0.6 * signal
    junk = 0.3 * rng.normal(size=n)
    assert st.clark_west(actual, small, good, lags=2)["cw_p"] < 0.01
    assert st.clark_west(actual, small, junk, lags=2)["cw_p"] > 0.05


# --------------------------------------------------------------------------- #
# The bond and the overlay
# --------------------------------------------------------------------------- #
def test_par_bond_duration_matches_closed_form():
    D, C = st.par_bond_duration_convexity(np.array([5.0]), maturity=20)
    v = 1 / 1.025
    assert D[0] == pytest.approx((1 - v ** 40) / 0.05, rel=1e-9)
    assert 12.0 < D[0] < 13.0 and C[0] > 100


def test_bond_return_is_carry_when_yields_do_not_move():
    idx = pd.date_range("1990-01-31", periods=24, freq=data.MONTH_END)
    y = pd.Series(6.0, index=idx)
    rf = pd.Series(0.004, index=idx)
    b = st.bond_returns(y, rf).dropna()
    np.testing.assert_allclose(b["bond"], 0.06 / 12)
    np.testing.assert_allclose(b["bond_ex"], 0.06 / 12 - 0.004)


def test_bond_return_falls_when_yields_rise():
    idx = pd.date_range("1990-01-31", periods=3, freq=data.MONTH_END)
    y = pd.Series([6.0, 7.0, 7.0], index=idx)
    b = st.bond_returns(y, pd.Series(0.0, index=idx))
    assert b["bond"].iloc[1] < -0.08          # ~11 years of duration × 100 bp


def test_timing_backtest_has_exactly_one_lag_and_charges_costs():
    idx = pd.date_range("2000-01-31", periods=6, freq=data.MONTH_END)
    bond = pd.DataFrame({"bond_ex": [0.01, 0.02, -0.01, 0.03, 0.0, 0.01]}, index=idx)
    w = pd.Series([1.5, 0.5, 1.5, 1.5, 0.5, 0.5], index=idx)
    bt = st.timing_backtest(bond, w, cost_bps=10.0)
    # the weight set at month 0 earns month 1
    assert bt["pos"].iloc[0] == 1.5 and bt.index[0] == idx[1]
    assert bt["gross"].iloc[0] == pytest.approx(1.5 * 0.02)
    assert bt["turnover"].iloc[1] == pytest.approx(1.0)
    assert bt["net"].iloc[1] == pytest.approx(0.5 * -0.01 - 10e-4 * 1.0)


def test_constant_weight_overlay_equals_the_benchmark():
    idx = pd.date_range("2000-01-31", periods=40, freq=data.MONTH_END)
    rng = np.random.default_rng(6)
    bond = pd.DataFrame({"bond_ex": rng.normal(0.002, 0.02, 40)}, index=idx)
    bt = st.timing_backtest(bond, pd.Series(1.0, index=idx), cost_bps=50.0)
    np.testing.assert_allclose(bt["net"], bt["const"])


def _synthetic_timer(world):
    df = world[0]
    lag = st.add_publication_lag(df)
    _, fc = st.oos_table(lag, "aaa", 12, 120, variables=("gap_exp",))
    bond = st.bond_returns(df["aaa"], df["rf"])
    g = st.timing_backtest(bond, st.timing_positions(fc["gap_exp"]), 5.0)
    y = st.timing_backtest(bond, st.timing_positions(fc["gap_exp"], "f_base"), 5.0)
    return st.timing_summary(g), st.hac_mean(g["net"] - y["net"])


def test_the_overlay_banks_a_planted_gibson_world(gibson_world):
    s, vs_yield = _synthetic_timer(gibson_world)
    assert s["diff_net_ann"] > 0 and s["diff_net_t"] > 2
    assert vs_yield["t"] > 2


def test_the_overlay_finds_nothing_in_the_fisher_world(fisher_world):
    s, vs_yield = _synthetic_timer(fisher_world)
    assert s["diff_net_t"] < 2
    assert vs_yield["t"] < 2


# --------------------------------------------------------------------------- #
# The verdict rules — both directions
# --------------------------------------------------------------------------- #
ALL = {"L1": True, "L2": True, "L3": True}
NONE = {"L1": False, "L2": False, "L3": False}


@pytest.mark.parametrize("us,de,stamp", [
    (ALL, ALL, "Real"),
    ({"L1": False, "L2": True, "L3": True}, {"L1": True, "L2": False, "L3": False}, "Real"),
    ({"L1": False, "L2": True, "L3": True}, NONE, "Mixed"),
    (NONE, {"L1": True, "L2": True, "L3": False}, "Mixed"),
    ({"L1": True, "L2": False, "L3": True}, NONE, "Weak"),   # L2 is required for "strong"
    (NONE, {"L1": False, "L2": False, "L3": True}, "Weak"),
    (NONE, NONE, "None"),
])
def test_signal_stamp_rule(us, de, stamp):
    assert st.signal_stamp(us, de) == stamp


def _timing(diff=0.02, t=2.5, net_sh=0.6, const_sh=0.5, subs=(0.01, 0.02), beats=True):
    return {"diff_net_ann": diff, "diff_net_t": t, "net": {"sharpe": net_sh},
            "const": {"sharpe": const_sh}, "beats_yield_only": beats,
            "subs": {"a": {"diff_net_ann": subs[0]}, "b": {"diff_net_ann": subs[1]}}}


def test_trad_stamp_investable_needs_everything():
    assert st.trad_stamp(_timing(), "Real") == "Investable"
    assert st.trad_stamp(_timing(), "Weak") == "Fragile"                    # no signal behind it
    assert st.trad_stamp(_timing(subs=(-0.001, 0.02)), "Real") == "Fragile"  # one regime only
    assert st.trad_stamp(_timing(beats=False), "Real") == "Fragile"         # yield did it
    assert st.trad_stamp(_timing(t=1.5), "Real") == "Fragile"


def test_trad_stamp_mirage():
    assert st.trad_stamp(_timing(diff=-0.01, t=-1.0), "Real") == "Mirage"
    assert st.trad_stamp(_timing(t=1.0, net_sh=0.4, const_sh=0.5), "Mixed") == "Mirage"


def _h(us, de, timing):
    base = {k: 0.1 for k in ("corr_logp", "corr_gap", "corr_pi", "corr_ewma", "t_gap", "t_pi",
                             "eg_p_gap", "eg_p_pi", "eg_p_ewma", "t_gap_vs_ewma", "cw_p_gap",
                             "cw_p_pi", "cw_p_gap_rol", "de_t_gap", "de_t_pi", "de_eg_p_gap",
                             "gi_corr_logp", "dis_corr_logp")}
    timing = {**timing, "turnover_ann": 0.1, "mean_pos": 1.2, "alpha_ann": 0.0, "alpha_t": 0.0,
              "diff_gross_ann": 0.01, "breakeven_bps": 100.0, "diff_vs_yield_only_ann": 0.01}
    timing["subs"] = {k: {**v, "t": 1.0} for k, v in timing["subs"].items()}
    return {**base, "ewma_hl": 36.0, "cost_bps": 5.0, "us_gibson": us, "de_gibson": de,
            "us_fisher": NONE, "de_fisher": NONE, "timing": timing}


def test_verdict_both_directions():
    hi = st.verdict(_h(ALL, ALL, _timing()))
    assert (hi["signal"], hi["trad"]) == ("Real", "Investable")
    lo = st.verdict(_h(NONE, NONE, _timing(diff=-0.01, t=-1.0)))
    assert (lo["signal"], lo["trad"]) == ("None", "Mirage")
    for v in (hi, lo):
        assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
        assert len(v["one_sentence"]) > 40
    assert "nothing reliable" in lo["one_sentence"]


# --------------------------------------------------------------------------- #
# Real tapes (gated) — the published facts, re-checked
# --------------------------------------------------------------------------- #
@needs_real
def test_real_us_gap_wins_differences_but_is_not_cointegrated():
    us = data.load_us_monthly()
    hr = st.diff_horse_race(us, "aaa", "gap_exp", "pi", lags=12)
    assert hr["t_gap"] > 2.0 and abs(hr["t_pi"]) < 2.0
    assert st.engle_granger(us["aaa"], us["gap_exp"])["eg_p"] > 0.05


@needs_real
def test_real_long_memory_fisher_is_cointegrated():
    us = data.load_us_monthly()
    e = st.adaptive_expectation(us["pi_1m"], 36)
    assert st.engle_granger(us["aaa"], e)["eg_p"] < 0.05


@needs_real
def test_real_germany_shows_nothing_in_differences():
    de = data.load_germany()
    hr = st.diff_horse_race(de, "R", "gap_exp", "pi", lags=4)
    assert abs(hr["t_gap"]) < 2.0 and abs(hr["t_pi"]) < 2.0


@needs_real
def test_real_raw_price_level_correlation_flips_sign_by_regime():
    us = data.load_us_monthly()
    gi = st.level_correlations(us, "aaa", ("logp",), *data.GREAT_INFLATION)["logp"]
    dis = st.level_correlations(us, "aaa", ("logp",), *data.DISINFLATION)["logp"]
    assert gi > 0.5 and dis < -0.5
