"""Strategy tests for Study 1022 — the Taylor rule, the real-time gaps, the inference, the trade.

Synthetic-first: every detector must fire on the planted world and stay quiet on the matched
null. Real-tape checks are gated on the bundled tapes being installed (they always are in CI).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from taylorgap import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch / statsmodels tapes not installed")


# --------------------------------------------------------------------------- #
# The rule and the gaps
# --------------------------------------------------------------------------- #
def test_taylor_rate_is_taylor_1993():
    pi = pd.Series([2.0, 4.0, 0.0])
    gap = pd.Series([0.0, 2.0, -4.0])
    i = st.taylor_rate(pi, gap)
    assert list(i) == [4.0, 8.0, -1.0]      # 2+2+0+0 ; 2+4+1+1 ; 2+0-1-2


def _toy_macro(n=90, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-03-31", periods=n, freq="QE-DEC")
    gdp = 2000 * np.exp(np.cumsum(0.008 + rng.normal(0, 0.008, n)))
    cpi = 30 * np.exp(np.cumsum(0.01 + rng.normal(0, 0.003, n)))
    u = 5 + np.cumsum(rng.normal(0, 0.2, n)) * 0.3
    tb = 4 + rng.normal(0, 1, n)
    return pd.DataFrame({"realgdp": gdp, "cpi": cpi, "unemp": u, "tbilrate": tb}, index=idx)


def test_one_sided_hp_never_looks_ahead():
    q = _toy_macro()
    lg = 100 * np.log(q["realgdp"])
    a = st.one_sided_hp_gap(lg)
    lg2 = lg.copy()
    lg2.iloc[70:] += 25.0                      # rewrite the future
    b = st.one_sided_hp_gap(lg2)
    assert np.allclose(a.iloc[:70].dropna(), b.iloc[:70].dropna())
    assert not np.allclose(a.iloc[70:], b.iloc[70:])


def test_one_sided_hp_endpoint_equals_full_filter_endpoint():
    q = _toy_macro()
    lg = 100 * np.log(q["realgdp"])
    a = st.one_sided_hp_gap(lg)
    assert a.iloc[-1] == pytest.approx(st.hp_cycle(lg.to_numpy())[-1])
    assert a.iloc[: st.HP_MIN_OBS - 1].isna().all()


def test_unemployment_gap_is_negative_when_unemployment_is_high():
    u = pd.Series([5.0] * 40 + [8.0])
    g = st.unemployment_gap(u)
    assert g.iloc[-1] < 0
    assert g.iloc[:39].isna().all()


def test_build_signals_applies_the_release_lag_exactly_once():
    q = _toy_macro()
    base = st.build_signals(q)
    q2 = q.copy()
    q2.iloc[60, q2.columns.get_loc("cpi")] *= 1.05      # a CPI shock dated quarter 60
    moved = st.build_signals(q2)
    assert moved["pi"].iloc[60] == pytest.approx(base["pi"].iloc[60])   # not yet released
    assert moved["pi"].iloc[61] != pytest.approx(base["pi"].iloc[61])   # released at t+1
    # the policy rate is a market price: not lagged
    q3 = q.copy()
    q3.iloc[60, q3.columns.get_loc("tbilrate")] += 1.0
    assert st.build_signals(q3)["i"].iloc[60] == pytest.approx(base["i"].iloc[60] + 1.0)


def test_tgap_is_rate_minus_rule():
    q = _toy_macro()
    s = st.build_signals(q).dropna()
    assert np.allclose(s["tgap_hp"], s["i"] - s["istar_hp"])
    assert np.allclose(s["istar_u"], st.taylor_rate(s["pi"], s["gap_u"]))


def test_demeaned_uses_only_the_past():
    x = pd.Series(np.arange(20.0), name="x")
    d = st.demeaned(x)
    x2 = x.copy()
    x2.iloc[15:] = 100.0
    assert np.allclose(d.iloc[:15].dropna(), st.demeaned(x2).iloc[:15].dropna())


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #
def test_forward_sum_aligns_the_target_to_the_signal():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert list(st.forward_sum(y, 1).iloc[:4]) == [2.0, 3.0, 4.0, 5.0]
    assert list(st.forward_sum(y, 2).iloc[:3]) == [5.0, 7.0, 9.0]


def test_nw_slope_recovers_a_slope_and_vectorises():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 500))
    y = 0.5 * x + rng.normal(size=(3, 500))
    v = st.nw_slope(y, x, 4)
    one = st.nw_slope(y[1], x[1], 4)
    assert np.allclose(v["slope"], 0.5, atol=0.12)
    assert v["t"][1] == pytest.approx(one["t"])


def test_hac_widens_the_error_under_overlap():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    e = pd.Series(rng.normal(size=403)).rolling(4).sum().dropna().to_numpy()
    y = e[:400]
    assert st.nw_slope(y, x, 8)["se"] > 0
    a = st.nw_slope(y, np.convolve(x, np.ones(4), "same"), 0)["se"]
    b = st.nw_slope(y, np.convolve(x, np.ones(4), "same"), 8)["se"]
    assert b > a


def test_predictive_regression_fires_on_the_plant(planted):
    p, truth = planted
    r = st.predictive_regression(p, "tgap", "eq_xs", 1)
    assert r["slope"] == pytest.approx(truth["beta_eq"], abs=0.006)
    assert r["t"] < -2.0
    assert st.predictive_regression(p, "tgap", "d_aaa", 1)["t"] < -2.0


def test_predictive_regression_is_quiet_on_the_null(null_world):
    p, _ = null_world
    assert abs(st.predictive_regression(p, "tgap", "eq_xs", 1)["t"]) < 2.0
    assert abs(st.predictive_regression(p, "tgap", "d_aaa", 1)["t"]) < 2.0


def test_stambaugh_bias_has_the_textbook_sign(null_world):
    """corr(u, v) < 0 -> gamma < 0 -> the OLS slope is biased UP -> the correction pulls it down."""
    p, _ = null_world
    s = st.stambaugh_correction(p, "tgap", "eq_xs")
    assert s["gamma"] < 0 and s["corr_uv"] < -0.2
    assert s["bias"] > 0 and s["slope_c"] < s["slope"]
    assert s["rho_c"] > s["rho"]


def test_simulation_null_reproduces_stambaugh_bias():
    """Under H0 with strongly correlated innovations the mean null slope is not zero."""
    p, _ = data.synthetic_quarterly(n_quarters=80, signal_strength=0.0, corr_uv=-0.9, rho=0.97)
    sm = st.simulation_null(p, "tgap", "eq_xs", horizons=(1,), n_sim=800)
    s = st.stambaugh_correction(p, "tgap", "eq_xs")
    assert sm[1]["null_mean_slope"] > 0
    assert np.sign(sm[1]["null_mean_slope"]) == np.sign(s["bias"])


def test_simulation_null_detects_the_plant_at_every_horizon(planted):
    p, _ = planted
    sm = st.simulation_null(p, "tgap", "eq_xs", n_sim=400)
    assert sm[1]["p_two"] < 0.05 and sm[4]["p_two"] < 0.05
    assert sm[1]["p_claim"] < 0.05


def test_simulation_null_is_quiet_on_the_null(null_world):
    p, _ = null_world
    sm = st.simulation_null(p, "tgap", "eq_xs", n_sim=400)
    assert all(sm[h]["p_two"] > 0.05 for h in st.HORIZONS)


def test_simulation_null_has_roughly_nominal_size():
    """Across independent null worlds the 5% test rejects about 5% of the time."""
    rej = 0
    n_worlds = 30
    for k in range(n_worlds):
        p, _ = data.synthetic_quarterly(n_quarters=160, signal_strength=0.0, seed=5000 + k)
        sm = st.simulation_null(p, "tgap", "eq_xs", horizons=(1, 8), n_sim=250, seed=k)
        rej += sm[8]["p_two"] < 0.05
    assert rej / n_worlds <= 0.17


def test_long_horizon_null_band_is_wider_than_the_naive_one(null_world):
    """Overlap fattens the t distribution; the simulated band shows it."""
    p, _ = null_world
    sm = st.simulation_null(p, "tgap", "eq_xs", n_sim=600)
    w1 = sm[1]["null_t_q"][2] - sm[1]["null_t_q"][0]
    w8 = sm[8]["null_t_q"][2] - sm[8]["null_t_q"][0]
    assert w8 > w1 > 3.0


def test_conditional_means_and_regime_split_run(planted):
    p, _ = planted
    c = st.conditional_means(p, "tgap", "eq_xs")
    assert c["mean_behind"] > c["mean_ahead"] and c["n_behind"] + c["n_ahead"] == len(p) - 1
    r = st.regime_split(p, "tgap", "eq_xs", break_date="1985-12-31")
    assert r["slope_pre"] < 0 and r["slope_post"] < 0 and abs(r["t_diff"]) < 2.5


# --------------------------------------------------------------------------- #
# The trade
# --------------------------------------------------------------------------- #
def test_timing_backtest_uses_exactly_one_lag(planted):
    p, _ = planted
    bt = st.timing_backtest(p, "tgap", 0.0)
    expect = (p["tgap"] < 0).astype(float).shift(1).reindex(bt.index)
    assert np.allclose(bt["pos"], expect)
    assert np.allclose(bt["strat_xs_gross"], bt["pos"] * p["eq_xs"].reindex(bt.index))


def test_costs_are_one_way_on_switches_only(planted):
    p, _ = planted
    a = st.timing_backtest(p, "tgap", 0.0)
    b = st.timing_backtest(p, "tgap", 50.0)
    diff = a["strat_xs_net"] - b["strat_xs_net"]
    assert np.allclose(diff, 0.005 * a["turnover"])
    assert set(np.unique(a["turnover"].iloc[1:])) <= {0.0, 1.0}


def test_timing_beats_buy_and_hold_on_the_plant_not_on_the_null(planted, null_world):
    sp = st.backtest_summary(st.timing_backtest(planted[0], "tgap", 10.0))
    sn = st.backtest_summary(st.timing_backtest(null_world[0], "tgap", 10.0))
    assert sp["sr_diff_net"] > 0.15 and sp["alpha_t"] > 1.5
    assert sn["sr_diff_net"] < 0.15


def test_breakeven_cost_is_zero_without_a_gross_edge(null_world):
    s = st.backtest_summary(st.timing_backtest(null_world[0], "tgap", 10.0))
    if s["sr_gross"] <= s["sr_bh"]:
        assert s["breakeven_bp"] == 0.0


def test_sharpe_diff_bootstrap_sees_a_dominant_strategy():
    rng = np.random.default_rng(3)
    b = rng.normal(0.01, 0.08, 200)
    a = b + 0.02 + rng.normal(0, 0.01, 200)
    r = st.sharpe_diff_bootstrap(a, b, n_boot=500)
    assert r["diff"] > 0 and r["p_one_sided"] < 0.05 and r["ci_lo"] > 0
    r0 = st.sharpe_diff_bootstrap(b, b, n_boot=200)
    assert r0["diff"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_panel_spans_1969_to_2009():
    p = st.real_panel()
    assert p.index[0] == pd.Timestamp("1969-03-31")
    assert p.index[-1] == pd.Timestamp(data.AS_OF)
    assert p.loc[:"2009-09-30", ["tgap_hp", "tgap_u"]].notna().all().all()


@needs_real
def test_real_gap_is_persistent_and_mostly_behind():
    p = st.real_panel()
    assert st.ar1(p["tgap_hp"].dropna().to_numpy())["rho"] > 0.7
    assert (p["tgap_hp"].dropna() < 0).mean() > 0.5


@needs_real
def test_real_realtime_gap_differs_from_hindsight():
    """Orphanides in miniature: the knowable gap is not the historian's gap."""
    ag = st.gap_agreement(st.real_panel())
    assert ag["corr_gap_hp_hind"] < 0.9
    assert ag["mean_abs_gap_rev"] > 0.5


@needs_real
def test_real_equity_leg_is_noise():
    """The pinned headline: no equity tailwind from being behind the curve."""
    p = st.real_panel()
    r = st.predictive_regression(p, "tgap_hp", "eq_xs", 1)
    assert abs(r["t"]) < 2.0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _h(**over):
    h = {"n_quarters": 163, "start": "1969Q1", "end": "2009Q3",
         "eq_primary": {"slope": -0.01, "t": -3.0, "slope_c": -0.01, "t_c": -2.8,
                        "p_sim": 0.01, "rho": 0.86, "corr_uv": -0.1},
         "eq_u_slope": -0.008, "eq_u_t": -2.1,
         "bond_primary": {"slope": -0.05, "t": -2.5, "p_sim": 0.02, "p_claim": 0.01},
         "specs": [{"label": "x", "slope": -0.01, "t": -3.0, "p_sim": 0.01}],
         "trade": {"share_invested": 0.6, "switches_per_year": 0.8, "sr_net": 0.6,
                   "sr_bh": 0.3, "sr_diff_net": 0.3, "ci_lo": 0.05, "ci_hi": 0.5,
                   "p_diff": 0.01, "ann_xs_net": 0.06, "ann_xs_bh": 0.05,
                   "alpha_ann": 0.03, "alpha_t": 2.5, "breakeven_bp": 300.0}}
    for k, v in over.items():
        if isinstance(v, dict) and k in h and isinstance(h[k], dict):
            h[k] = {**h[k], **v}
        else:
            h[k] = v
    return h


def test_verdict_real_and_investable_when_everything_clears():
    v = st.verdict(_h())
    assert v["signal"] == "Real" and v["trad"] == "Investable"
    assert set(v) >= {"signal", "signal_why", "trad", "trad_why", "one_sentence"}


def test_verdict_mixed_when_one_leg_fails():
    v = st.verdict(_h(bond_primary={"slope": 0.01, "t": 0.5, "p_sim": 0.6}))
    assert v["signal"] == "Mixed"
    v2 = st.verdict(_h(eq_primary={"p_sim": 0.3, "t_c": -1.0}))
    assert v2["signal"] == "Mixed"


def test_verdict_equity_leg_needs_the_unemployment_gap_to_agree():
    v = st.verdict(_h(eq_u_slope=0.002, bond_primary={"slope": 0.01, "p_sim": 0.6}))
    assert v["signal"] != "Real" and not v["eq_real"]


def test_verdict_weak_and_none():
    weak = _h(eq_primary={"slope": -0.004, "slope_c": -0.004, "t_c": -1.5, "p_sim": 0.08},
              bond_primary={"slope": -0.02, "t": -1.5, "p_sim": 0.15},
              specs=[{"label": "x", "slope": -0.004, "t": -1.5, "p_sim": 0.08}])
    assert st.verdict(weak)["signal"] == "Weak"
    none = _h(eq_primary={"slope": 0.001, "slope_c": 0.001, "t_c": 0.4, "p_sim": 0.7},
              bond_primary={"slope": -0.03, "t": -2.0, "p_sim": 0.06},
              specs=[{"label": "x", "slope": -0.03, "t": -2.0, "p_sim": 0.06},
                     {"label": "y", "slope": 0.001, "t": 0.4, "p_sim": 0.7}])
    assert st.verdict(none)["signal"] == "None"


def test_verdict_a_significant_wrong_sign_is_not_support():
    v = st.verdict(_h(eq_primary={"slope": 0.01, "slope_c": 0.01, "t_c": 3.0, "p_sim": 0.01},
                      bond_primary={"slope": 0.05, "t": 3.0, "p_sim": 0.01},
                      specs=[{"label": "x", "slope": 0.01, "t": 3.0, "p_sim": 0.01}]))
    assert v["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_h(trade={"p_diff": 0.2}))["trad"] == "Fragile"
    assert st.verdict(_h(trade={"sr_diff_net": 0.05}))["trad"] == "Fragile"
    assert st.verdict(_h(trade={"sr_diff_net": -0.1}))["trad"] == "Mirage"
    none = _h(eq_primary={"slope": 0.001, "slope_c": 0.001, "t_c": 0.4, "p_sim": 0.7},
              bond_primary={"slope": 0.01, "t": 0.3, "p_sim": 0.7},
              specs=[{"label": "x", "slope": 0.001, "t": 0.4, "p_sim": 0.7}])
    assert st.verdict(none)["trad"] == "Mirage"        # a lucky rule on no signal is a mirage


def test_verdict_prose_names_orphanides_and_the_race():
    v = st.verdict(_h(eq_primary={"slope": 0.001, "slope_c": 0.001, "t_c": 0.4, "p_sim": 0.7}))
    assert "Orphanides" in v["signal_why"] and "final-vintage" in v["signal_why"]
    assert "buy-and-hold" in v["trad_why"] and "10 bp" in v["trad_why"]
