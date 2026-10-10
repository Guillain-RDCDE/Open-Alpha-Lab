"""Strategy tests for Study 1017 — dating, duration model, nulls, prediction, rule, verdict."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oldage import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover logic")


def _level(vals, start="2000-01-31"):
    idx = pd.date_range(start, periods=len(vals), freq=pd.offsets.MonthEnd())
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


# --------------------------------------------------------------------------- #
# Dating
# --------------------------------------------------------------------------- #
# 100 → dips to 95 (trough, i=2) → climbs to 130 (peak, i=6) → falls to 100 (−23%, i=8)
# → trough 98 (i=9) → rises 20%+ to 120 (i=11)
TOY = [100, 97, 95, 105, 115, 120, 130, 115, 100, 98, 110, 120, 125]


def test_turning_points_on_a_toy_path():
    tps = st.turning_points(np.array(TOY), 0.20)
    assert [(i, k) for i, k, _ in tps] == [(2, "trough"), (6, "peak"), (9, "trough")]
    # each turning point is confirmed only once the 20% move has happened
    assert [c for _, _, c in tps] == [4, 8, 11]     # 115 ≥ 95 × 1.2 at i=4


def test_date_cycles_durations_and_censoring():
    cy = st.date_cycles(_level(TOY), 0.20)
    assert list(cy["phase"]) == ["bull", "bear", "bull"]
    assert list(cy["duration"]) == [4, 3, 3]
    assert list(cy["censored"]) == [False, False, True]
    assert cy["amplitude"].iloc[0] == pytest.approx(130 / 95 - 1)


def test_sample_start_is_not_a_trough():
    """A bull that starts on the first observation has an unknown start and is dropped."""
    tps = st.turning_points(np.array([100, 110, 125, 130, 100, 90, 110]), 0.20)
    assert tps[0][1] == "peak" and tps[0][0] == 3


def test_asymmetric_filter_ends_bulls_sooner():
    path = np.array([100, 80, 100, 120, 105, 100, 125])     # −16.7% dip from 120
    sym = st.turning_points(path, 0.20, 0.20)
    asym = st.turning_points(path, 0.20, 0.15)
    assert len(asym) > len(sym)


def test_bull_durations_match_date_cycles():
    lv = _level(TOY)
    d, c = st.bull_durations(lv, 0.20)
    cy = st.date_cycles(lv, 0.20)
    b = cy[cy["phase"] == "bull"]
    assert list(d) == list(b["duration"]) and list(c) == list(b["censored"])


# --------------------------------------------------------------------------- #
# Duration model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k_true", [0.8, 1.0, 2.5])
def test_weibull_mle_recovers_the_shape(k_true):
    rng = np.random.default_rng(7)
    d = 40.0 * rng.weibull(k_true, 800)
    f = st.fit_weibull(d)
    assert f["k"] == pytest.approx(k_true, rel=0.08)
    assert f["lam"] == pytest.approx(40.0, rel=0.08)


def test_weibull_handles_right_censoring():
    """Censoring at a fixed horizon must not bias the shape (naively dropping would)."""
    rng = np.random.default_rng(11)
    d = 40.0 * rng.weibull(2.0, 1500)
    cens = d > 55
    d_obs = np.minimum(d, 55)
    f = st.fit_weibull(d_obs, cens)
    assert f["k"] == pytest.approx(2.0, rel=0.1)
    assert f["n_censored"] == int(cens.sum())


def test_weibull_lr_test_separates_exponential_from_ageing():
    rng = np.random.default_rng(3)
    assert st.fit_weibull(30 * rng.exponential(1.0, 400))["p_k1"] > 0.01
    assert st.fit_weibull(30 * rng.weibull(2.0, 400))["p_k1"] < 1e-6


def test_weibull_declines_with_too_few_spells():
    assert st.fit_weibull([10.0, 20.0]) == {}
    assert st.fit_weibull([10.0, 20.0, 30.0, 40.0], [True, True, False, False]) == {}


def test_weibull_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(5)
    d = 30 * rng.weibull(1.5, 60)
    f = st.fit_weibull(d)
    b = st.weibull_bootstrap(d, n_boot=200, seed=1)
    assert b["k_lo"] < f["k"] < b["k_hi"]


def test_life_table_is_flat_for_exponential_spells():
    rng = np.random.default_rng(9)
    d = rng.exponential(36.0, 20000)
    lt = st.life_table(d)
    expected = 1 - np.exp(-12 / 36.0)
    assert np.allclose(lt["annual_hazard"].to_numpy(), expected, atol=0.02)
    assert lt["at_risk"].iloc[0] == 20000


# --------------------------------------------------------------------------- #
# Real-time state — no look-ahead
# --------------------------------------------------------------------------- #
def test_realtime_state_is_causal(null_world):
    df, _ = null_world
    lv = data.total_return_index(df["mkt"])
    full = st.realtime_state(lv)
    cut = 600
    part = st.realtime_state(lv.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_realtime_age_agrees_with_ex_post_dating(null_world):
    df, _ = null_world
    lv = data.total_return_index(df["mkt"])
    rt = st.realtime_state(lv)
    cy = st.date_cycles(lv)
    for b in cy[(cy["phase"] == "bull") & ~cy["censored"]].itertuples():
        conf = lv.index.get_loc(b.confirmed)
        # on the bar the peak is confirmed, the completed length equals the ex-post duration
        assert rt["completed_bull"].iloc[conf] == b.duration
        # one bar earlier the bull was still alive in real time, aged from its trough
        assert rt["age"].iloc[conf - 1] == conf - 1 - b.start_i


def test_toy_realtime_state():
    rt = st.realtime_state(_level(TOY))
    assert list(rt["state"]) == [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1]
    assert rt["age"].iloc[4] == 2 and rt["age"].iloc[12] == 3
    assert rt["age"].iloc[:4].isna().all() and rt["age"].iloc[8:11].isna().all()
    assert rt["completed_bull"].iloc[8] == 4


def test_forward_return_alignment():
    r = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
    f = st.forward_log_return(r, None, h=2)
    assert f.iloc[0] == pytest.approx(np.log(1.1) + np.log(1.2))
    assert np.isnan(f.iloc[3]) and np.isnan(f.iloc[4])


def test_forward_drawdown():
    lv = pd.Series([100.0, 110.0, 88.0, 99.0, 120.0])
    dd = st.forward_drawdown(lv, h=3)
    assert dd.iloc[0] == pytest.approx(88 / 110 - 1)


def test_ols_hac_recovers_slope_and_widens_under_overlap():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=1200)) / 30
    e = pd.Series(rng.normal(size=1212)).rolling(12).sum().dropna().to_numpy()[:1200]
    y = 0.5 * x + e
    a = st.ols_hac(y, x, lags=0)
    b = st.ols_hac(y, x, lags=18)
    assert a["slope"] == pytest.approx(b["slope"])
    assert b["se"] > a["se"]


# --------------------------------------------------------------------------- #
# Nulls
# --------------------------------------------------------------------------- #
def test_null_simulators_match_moments():
    rw = st.simulate_log_returns("rw", 2000, 50, {"mu": 0.008, "sigma": 0.05}, seed=1)
    assert rw.shape == (50, 2000)
    assert rw.mean() == pytest.approx(0.008, abs=0.001)
    assert rw.std() == pytest.approx(0.05, rel=0.03)
    gp = {"mu": 0.008, "omega": 1e-4, "alpha": 0.12, "beta": 0.84, "nu": 7.0}
    g = st.simulate_log_returns("garch", 2000, 50, gp, seed=1)
    assert g.std() == pytest.approx(np.sqrt(1e-4 / (1 - 0.96)), rel=0.15)
    with pytest.raises(ValueError):
        st.simulate_log_returns("ar1", 10, 2, gp)


def test_the_dating_rule_manufactures_ageing_on_random_walks():
    """The study's central mechanical fact: memoryless paths dated ±20% show k > 1."""
    paths = st.simulate_log_returns("rw", 1109, 120, {"mu": 0.0079, "sigma": 0.053}, seed=2)
    ks = [st.path_stats(p, with_prediction=False)["k"] for p in paths]
    assert np.nanmedian(ks) > 1.1
    assert np.mean(np.asarray(ks) > 1) > 0.7


def test_null_p_bounds():
    v = np.arange(99.0)
    assert st.null_p(1000.0, v, "greater") == pytest.approx(1 / 100)
    assert st.null_p(-1.0, v, "greater") == 1.0
    assert st.null_p(-1.0, v, "less") == pytest.approx(1 / 100)


def test_detector_fires_on_planted_ageing(planted):
    df, truth = planted
    assert truth["n_crash"] > 0
    res = st.duration_test(df["mkt"], n_sims=100, n_boot=50, kinds=("rw",), rf=df["rf"])
    n = res["nulls"]["rw"]
    assert res["fit"]["k"] > n["k_null_hi"]
    assert n["p_k"] < 0.05
    assert res["pred_raw"]["t"] < -2 and n["p_pred"] < 0.05


def test_detector_stays_quiet_on_the_null(null_world):
    df, _ = null_world
    res = st.duration_test(df["mkt"], n_sims=100, n_boot=50, kinds=("rw",), rf=df["rf"])
    n = res["nulls"]["rw"]
    assert n["p_k"] > 0.05
    assert n["p_pred"] > 0.05
    # …even though the naive Weibull shape on this memoryless path exceeds 1
    assert res["fit"]["k"] > 1.0


def test_garch_null_runs_with_supplied_params(null_world):
    df, _ = null_world
    gp = {"mu": 0.0, "omega": 1e-4, "alpha": 0.1, "beta": 0.85, "nu": 8.0}
    nd = st.null_distribution(df["mkt"], "garch", n_sims=20, garch_params=gp,
                              with_prediction=False)
    assert nd["params"]["mu"] == pytest.approx(np.log1p(df["mkt"]).mean())   # tape drift
    assert len(nd["table"]) == 20 and nd["table"]["k"].notna().all()


# --------------------------------------------------------------------------- #
# The de-risking rule
# --------------------------------------------------------------------------- #
def test_backtest_has_exactly_one_lag_and_no_cost_when_static():
    idx = pd.date_range("2000-01-31", periods=6, freq=pd.offsets.MonthEnd())
    r = pd.Series([0.01, 0.02, -0.03, 0.04, 0.01, 0.0], index=idx)
    rf = pd.Series(0.001, index=idx)
    w = pd.Series([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], index=idx)
    bt = st.age_rule_backtest(r, rf, w, cost_bps=0.0)
    # the 0.5 set at the close of month 2 earns month 3
    assert bt["weight"].tolist() == [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
    assert bt["gross"].iloc[3] == pytest.approx(0.5 * 0.04 + 0.5 * 0.001)
    bh = st.age_rule_backtest(r, rf, pd.Series(1.0, index=idx), cost_bps=25.0)
    assert np.allclose(bh["net"], r) and bh["turnover"].sum() == 0


def test_backtest_charges_costs_on_traded_nav():
    idx = pd.date_range("2000-01-31", periods=4, freq=pd.offsets.MonthEnd())
    r = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    rf = pd.Series(0.0, index=idx)
    w = pd.Series([1.0, 0.5, 0.5, 0.5], index=idx)
    bt = st.age_rule_backtest(r, rf, w, cost_bps=10.0)
    assert bt["turnover"].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0])
    assert bt["net"].iloc[2] == pytest.approx(-0.5 * 10 / 1e4)


def test_rule_weights_use_only_known_bulls(planted):
    df, _ = planted
    lv = data.total_return_index(df["mkt"])
    W = st.age_rule_weights(lv)
    first = W["median_age"].first_valid_index()
    assert (W.loc[:first, "weight"].iloc[:-1] == 1.0).all()
    # the median only changes on bars where a bull's peak is confirmed
    rt = st.realtime_state(lv)
    changes = W["median_age"].diff().fillna(0).ne(0)
    assert rt.loc[changes, "completed_bull"].notna().all()
    assert set(W["weight"].unique()) <= {0.5, 1.0}
    W2 = st.age_rule_weights(lv, prior_durations=[10.0, 20.0, 30.0])
    assert W2["median_age"].iloc[0] == 20.0


def test_sharpe_diff_bootstrap():
    rng = np.random.default_rng(4)
    b = rng.normal(0.005, 0.04, 1200)
    same = st.sharpe_diff_bootstrap(b, b, n_boot=200)
    assert same["diff"] == 0.0
    better = st.sharpe_diff_bootstrap(b + 0.004, b, n_boot=300)
    assert better["diff"] > 0 and better["p"] < 0.05


def test_planted_world_rewards_de_risking_old_bulls(planted):
    """Machinery check for beat 6: where bulls truly die of old age, the rule should not lose."""
    df, _ = planted
    lv = data.total_return_index(df["mkt"])
    W = st.age_rule_weights(lv)
    bt = st.age_rule_backtest(df["mkt"], df["rf"], W["weight"], 10.0)
    s = st.summarize_backtest(bt)
    assert s["sharpe_net"] > s["sharpe_bh"]


# --------------------------------------------------------------------------- #
# Verdict — both directions, thresholds fixed in advance
# --------------------------------------------------------------------------- #
def _h(**kw):
    base = dict(n_years=92.0, n_bulls=11, k=1.1, k_lo=0.7, k_hi=2.0, k_null_rw=1.22,
                k_null_garch=0.91, p_k_rw=0.72, p_k_garch=0.24, p_pred_rw=0.35,
                p_pred_garch=0.33, pred_t=-0.99, pred_slope=-0.004, cost_bps=10.0,
                sharpe_bh=0.57, sharpe_net=0.49, sharpe_diff=-0.08, sharpe_lo=-0.16,
                sharpe_hi=0.01, p_sharpe=0.97, tw_bh=4787.0, tw_net=773.0,
                share_derisked=0.59, sp_sharpe_diff=-0.09)
    base.update(kw)
    return base


def test_verdict_none_mirage_on_a_null_result():
    v = st.verdict(_h())
    assert v["signal"] == "None" and v["trad"] == "Mirage"
    assert "artefact" in v["signal_why"]


def test_verdict_real_needs_both_nulls():
    assert st.verdict(_h(p_k_rw=0.01, p_k_garch=0.02))["signal"] == "Real"
    # beating the random walk but not GARCH is not enough
    assert st.verdict(_h(p_k_rw=0.01, p_k_garch=0.30))["signal"] == "None"
    assert st.verdict(_h(p_k_rw=0.01, p_k_garch=0.10))["signal"] == "Weak"


def test_verdict_prediction_route_needs_hac_and_simulation():
    assert st.verdict(_h(pred_t=-2.5, p_pred_rw=0.01, p_pred_garch=0.02))["signal"] == "Real"
    # a HAC t below −2 that random walks also produce reads Weak at best
    assert st.verdict(_h(pred_t=-2.5, p_pred_rw=0.10, p_pred_garch=0.12))["signal"] == "Weak"
    assert st.verdict(_h(pred_t=-1.5, p_pred_rw=0.01, p_pred_garch=0.01))["signal"] == "Weak"


def test_verdict_naive_k_above_one_earns_nothing():
    assert st.verdict(_h(k=1.6, k_lo=1.1, k_hi=2.4))["signal"] == "None"


def test_verdict_tradability_ladder():
    real = dict(p_k_rw=0.01, p_k_garch=0.01)
    assert st.verdict(_h(**real, sharpe_diff=0.10, p_sharpe=0.01,
                         sp_sharpe_diff=0.05))["trad"] == "Investable"
    # Investable needs the out-of-sample tape too
    assert st.verdict(_h(**real, sharpe_diff=0.10, p_sharpe=0.01,
                         sp_sharpe_diff=-0.05))["trad"] == "Fragile"
    assert st.verdict(_h(sharpe_diff=0.05, p_sharpe=0.15))["trad"] == "Fragile"
    assert st.verdict(_h(sharpe_diff=0.05, p_sharpe=0.40))["trad"] == "Mirage"


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_ff_bull_markets():
    ff = data.load_ff_monthly()
    lv = data.total_return_index(ff["mkt"])
    cy = st.date_cycles(lv, 0.20)
    b = cy[cy["phase"] == "bull"]
    assert int((~b["censored"]).sum()) == 11
    last = b.iloc[-1]
    assert last["censored"] and last["start"] == pd.Timestamp("2009-02-28")
    assert pd.Timestamp("1987-08-31") in set(b["end"])        # the 1987 peak
    f = st.fit_weibull(*st.bull_durations(lv, 0.20))
    assert 0.8 < f["k"] < 1.5


@needs_real
def test_real_sp_daily_dates_the_2020_crash():
    sp = data.load_sp500_daily()
    cy = st.date_cycles(sp, 0.20)
    peaks = set(cy.loc[cy["phase"] == "bull", "end"])
    assert pd.Timestamp("2020-02-19") in peaks
    assert pd.Timestamp("2007-10-09") in peaks
