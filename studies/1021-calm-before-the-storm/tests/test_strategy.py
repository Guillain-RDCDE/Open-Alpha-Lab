"""Strategy tests for Study 1021 — does prolonged calm breed the next crash?"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from calm import data, strategy as st  # noqa: E402

needs_ff = pytest.mark.skipif(not data.have_ff(), reason="arch Fama-French tape absent")


def _null(world, kind="garch", n_paths=150, **kw):
    f, _ = world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.mean_reversion_null(f, kind, st.PRIMARY_H, n_paths, seed=5, **kw)


# --------------------------------------------------------------------------- #
# The signal
# --------------------------------------------------------------------------- #
def test_calm_signal_is_past_only(planted):
    """Rewriting the future must not move a single past reading."""
    f, _ = planted
    r = f["mkt"].copy()
    a = st.calm_signal(r)["calm"]
    r2 = r.copy()
    r2.iloc[1500:] = np.random.default_rng(9).normal(0.0, 0.3, len(r2) - 1500)
    b = st.calm_signal(r2)["calm"]
    pd.testing.assert_series_equal(a.iloc[:1500], b.iloc[:1500])


def test_calm_signal_warm_up_is_declared_not_filled(planted):
    f, _ = planted
    c = st.calm_signal(f["mkt"])["calm"]
    first = int(np.argmax(c.notna().to_numpy()))
    assert first == st.VOL_WINDOW + st.TREND_WINDOW + st.CALM_WINDOW - 3
    assert c.iloc[:first].isna().all()


def test_calm_is_zero_when_vol_is_above_trend():
    """High vol is not 'negative calm' — the claim is about quiet only."""
    rng = np.random.default_rng(0)
    r = pd.Series(np.concatenate([rng.normal(0, 0.02, 200), rng.normal(0, 0.08, 100)]),
                  index=pd.date_range("1950-01-31", periods=300, freq="ME"))
    s = st.calm_signal(r)
    assert (s["low"].dropna() >= 0).all()
    assert s["low"].iloc[215:260].max() == 0.0


# --------------------------------------------------------------------------- #
# The outcomes
# --------------------------------------------------------------------------- #
def test_forward_drawdown_on_a_hand_example():
    r = pd.Series([0.0, 0.10, -0.20, 0.05, -0.10, 0.30],
                  index=pd.date_range("2000-01-31", periods=6, freq="ME"))
    fo = st.forward_outcomes(r, None, H=3)
    # from the close of month 0: wealth 1.10, 0.88, 0.924 -> peak 1.10, trough 0.88
    assert fo["mdd"].iloc[0] == pytest.approx(0.20, abs=1e-12)
    # from the close of month 1: 0.8, 0.84, 0.756 -> peak 1 (start), trough 0.756
    assert fo["mdd"].iloc[1] == pytest.approx(1 - 0.756, abs=1e-12)
    assert fo["crash"].iloc[1] == 1.0 and fo["crash"].iloc[0] == 1.0
    assert fo["mdd"].iloc[-3:].isna().all()


def test_forward_window_excludes_the_signal_month():
    """A crash IN month t must not count as 'after' the signal known at its close."""
    r = pd.Series([-0.5, 0.01, 0.01, 0.01], index=pd.date_range("2000-01-31", periods=4,
                                                              freq="ME"))
    fo = st.forward_outcomes(r, None, H=2)
    assert fo["mdd"].iloc[0] == 0.0


def test_forward_vol_and_return():
    r = pd.Series([0.0, 0.02, -0.02, 0.02, -0.02],
                  index=pd.date_range("2000-01-31", periods=5, freq="ME"))
    rf = pd.Series(0.0, index=r.index)
    fo = st.forward_outcomes(r, rf, H=4)
    assert fo["vol"].iloc[0] == pytest.approx(0.02 * np.sqrt(12))
    assert fo["ret"].iloc[0] == pytest.approx(2 * np.log(1.02) + 2 * np.log(0.98))


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def test_hac_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)
    y = 0.5 + 0.3 * x + rng.normal(size=3000)
    r = st.hac_slope(y, x, lags=5)
    assert r["slope"] == pytest.approx(0.3, abs=0.05)
    assert r["t"] > 10


def test_hac_widens_the_error_under_overlap():
    """Overlapping windows make plain OLS overconfident — which is why HAC is the default."""
    rng = np.random.default_rng(2)
    e = rng.normal(size=3000)
    x = pd.Series(rng.normal(size=3000)).rolling(24).mean().to_numpy()
    y = pd.Series(e).rolling(24).sum().to_numpy()
    a = st.hac_slope(y, x, lags=0)
    b = st.hac_slope(y, x, lags=48)
    assert b["se"] > 2 * a["se"]


def test_predictive_regression_reports_the_nonoverlapping_check(planted):
    f, _ = planted
    sig = st.calm_signal(f["mkt"])
    fo = st.forward_outcomes(f["mkt"], f["rf"], 24)
    r = st.predictive_regression(sig["calm"], fo["mdd"], 24)
    assert np.isfinite(r["t_nonoverlap_median"])
    assert r["n_nonoverlap"] == r["n"] // 24


def test_wilson_interval_brackets_the_share():
    lo, hi = st.wilson(10, 100)
    assert lo < 0.10 < hi
    assert st.wilson(0, 20)[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Positive and negative controls — the planted world vs the matched null
# --------------------------------------------------------------------------- #
def test_the_detector_fires_on_the_planted_world(planted):
    f, _ = planted
    sig = st.calm_signal(f["mkt"])
    fo = st.forward_outcomes(f["mkt"], f["rf"], 24)
    r = st.predictive_regression(sig["calm"], fo["mdd"], 24)
    assert r["slope"] > 0 and r["t"] > 2.0
    m = _null(planted)
    assert m["mdd"]["p_value"] < 0.05
    assert m["mdd"]["real"] > m["mdd"]["null_q95"]


def test_the_detector_stays_quiet_on_the_null_world(null_world):
    f, _ = null_world
    sig = st.calm_signal(f["mkt"])
    fo = st.forward_outcomes(f["mkt"], f["rf"], 24)
    r = st.predictive_regression(sig["calm"], fo["mdd"], 24)
    assert r["t"] < 2.0
    m = _null(null_world)
    assert m["mdd"]["p_value"] > 0.10


def test_mean_reversion_alone_makes_calm_predict_SMALLER_drawdowns(null_world):
    """The confound, signed: with no feedback, calm is followed by calm, not by storm."""
    m = _null(null_world)
    assert m["mdd"]["null_mean"] < 0
    assert m["vol"]["null_mean"] < 0


def test_the_figarch_null_runs_and_is_calibrated(null_world):
    m = _null(null_world, kind="figarch", n_paths=40)
    assert 0 < m["mdd"]["p_value"] <= 1
    assert 0 < m["params"]["d"] < 1


def test_garch_simulation_has_the_fitted_persistence(null_world):
    f, _ = null_world
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.fit_vol_model(f["mkt_rf"], "garch")
    p = res.params
    assert 0.85 < p["alpha[1]"] + p["beta[1]"] < 1.0
    sims = st.simulate_null(res, "garch", 600, 8, seed=3)
    assert sims.shape == (600, 8)
    a = np.abs(sims - sims.mean(axis=0))
    ac = np.mean([np.corrcoef(a[1:, j], a[:-1, j])[0, 1] for j in range(8)])
    assert ac > 0.05


# --------------------------------------------------------------------------- #
# The two horizons
# --------------------------------------------------------------------------- #
def test_low_vol_now_predicts_low_vol_next_months(null_world):
    """The short-horizon sign: clustering. Calm now, calm soon."""
    f, _ = null_world
    sig = st.calm_signal(f["mkt"])
    p = st.vol_horizon_profile(f["mkt"], -sig["dev"], lags=(0, 3))
    assert (p["corr"] < -0.1).all()


def test_clustering_ar1_recovers_a_known_persistence():
    rng = np.random.default_rng(4)
    x = np.zeros(2000)
    for t in range(1, 2000):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.3)
    rv = pd.Series(np.exp(x - 2), index=pd.date_range("1900-01-31", periods=2000, freq="ME"))
    a = st.clustering_ar1(rv)
    assert a["phi"] == pytest.approx(0.7, abs=0.05)
    assert a["half_life_months"] == pytest.approx(np.log(0.5) / np.log(0.7), rel=0.25)


# --------------------------------------------------------------------------- #
# The trade
# --------------------------------------------------------------------------- #
def _ff(f):
    return f[["mkt", "rf", "mkt_rf"]]


def test_backtest_applies_exactly_one_lag():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    ff = pd.DataFrame({"mkt": [0.10, 0.20, -0.10, 0.05], "rf": 0.0, "mkt_rf": 0.0},
                      index=idx)
    w = pd.Series([1.0, 0.0, 1.0, 1.0], index=idx)
    bt = st.backtest(ff, w, cost_bps=0)
    # weight decided at close of Jan (1.0) earns Feb; decided at Feb (0.0) earns Mar
    assert bt.loc[idx[1], "gross"] == pytest.approx(0.20)
    assert bt.loc[idx[2], "gross"] == pytest.approx(0.0)
    assert bt.loc[idx[3], "gross"] == pytest.approx(0.05)


def test_buy_and_hold_pays_no_costs(planted):
    ff = _ff(planted[0])
    bt = st.backtest(ff, pd.Series(1.0, index=ff.index), cost_bps=50)
    assert np.allclose(bt["net"], bt["gross"])
    assert np.allclose(bt["gross"], ff["mkt"].iloc[1:])


def test_costs_bite_on_turnover(planted):
    ff = _ff(planted[0])
    w = pd.Series(np.where(np.arange(len(ff)) % 2 == 0, 1.0, 0.0), index=ff.index)
    a = st.summary(st.backtest(ff, w, 0))
    b = st.summary(st.backtest(ff, w, 25))
    assert b["sharpe_net"] < a["sharpe_net"]
    assert a["sharpe_gross"] == pytest.approx(b["sharpe_gross"])
    assert a["turnover_yr"] > 10


def test_calm_derisk_weights_are_two_valued_and_past_only(planted):
    f, _ = planted
    w = st.rule_weights(f["mkt"], "calm_derisk")
    assert set(w.dropna().unique()) <= {0.5, 1.0}
    r2 = f["mkt"].copy()
    r2.iloc[2000:] = np.random.default_rng(9).normal(-0.05, 0.2, len(r2) - 2000)
    w2 = st.rule_weights(r2, "calm_derisk")
    pd.testing.assert_series_equal(w.iloc[:2000], w2.iloc[:2000])


def test_vol_target_is_unlevered(planted):
    w = st.rule_weights(planted[0]["mkt"], "vol_target").dropna()
    assert (w <= 1.0 + 1e-12).all() and (w > 0).all()


def test_derisking_on_calm_helps_in_the_planted_world_drawdown(planted):
    ff = _ff(planted[0])
    w = st.rule_weights(ff["mkt"], "calm_derisk")
    start = w.first_valid_index()
    a = st.summary(st.backtest(ff.loc[start:], w.loc[start:]))
    b = st.summary(st.backtest(ff.loc[start:], pd.Series(1.0, index=ff.loc[start:].index)))
    assert a["max_dd"] >= b["max_dd"] - 0.02


def test_sharpe_bootstrap_of_a_series_against_itself_is_zero():
    rng = np.random.default_rng(6)
    x = pd.Series(rng.normal(0.005, 0.04, 600))
    b = st.sharpe_diff_bootstrap(x, x, n_boot=200)
    assert b["diff"] == 0.0 and b["ci"] == (0.0, 0.0)


def test_cost_sweep_is_monotone(planted):
    ff = _ff(planted[0])
    w = st.rule_weights(ff["mkt"], "calm_derisk")
    s = st.cost_sweep(ff, w, costs=(0, 25, 100))
    assert s["sharpe_rule"].is_monotonic_decreasing
    assert s["sharpe_bh"].nunique() == 1


# --------------------------------------------------------------------------- #
# Real tape (always available: arch ships the Fama-French file)
# --------------------------------------------------------------------------- #
@needs_ff
def test_real_primary_regression_runs_and_is_finite():
    ff = data.load_ff()
    tab = st.horizon_table(ff["mkt"], ff["rf"], horizons=(24,))
    row = tab[tab["outcome"] == "mdd"].iloc[0]
    assert np.isfinite(row["slope"]) and np.isfinite(row["t_hac"])
    assert row["n"] > 800


@needs_ff
def test_real_garch_fit_is_persistent():
    ff = data.load_ff()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.fit_vol_model(ff["mkt_rf"], "garch")
    assert 0.9 < res.params["alpha[1]"] + res.params["beta[1]"] < 1.0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"primary_t": 2.6, "primary_slope": 0.03, "primary_t_nonoverlap": 2.1,
         "n_indep": 37, "p_garch": 0.01, "p_figarch": 0.02, "p_garch_vol": 0.03,
         "null_garch_mean": -0.013, "half_slopes": [0.02, 0.04], "ar1_phi": 0.69,
         "ar1_t": 14.0, "n_episodes": 7, "derisk_sharpe_diff": 0.10, "derisk_p": 0.01,
         "derisk_max_dd": -0.30, "bh_max_dd": -0.50, "derisk_sharpe": 0.60,
         "bh_sharpe": 0.50, "voltarget_sharpe": 0.55, "derisk_terminal": 900.0,
         "bh_terminal": 800.0, "bt_years": 67.0, "derisk_turnover": 0.05}
    h.update(over)
    return h


def test_verdict_signal_branches():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(half_slopes=[-0.01, 0.04]))["signal"] == "Mixed"
    assert st.verdict(_headline(p_figarch=0.20))["signal"] == "Weak"      # raw t, null absorbs
    assert st.verdict(_headline(primary_t=0.9, p_garch=0.08, p_figarch=0.06))["signal"] \
        == "Weak"
    assert st.verdict(_headline(primary_t=0.9, p_garch=0.30, p_figarch=0.40))["signal"] \
        == "None"


def test_verdict_real_needs_the_raw_t_too():
    """Beating the null with a sub-2 t is not enough — the inference bar is on the tape."""
    assert st.verdict(_headline(primary_t=1.9))["signal"] != "Real"


def test_verdict_tradability_branches():
    assert st.verdict(_headline())["trad"] == "Investable"
    assert st.verdict(_headline(derisk_p=0.30))["trad"] == "Fragile"
    assert st.verdict(_headline(voltarget_sharpe=0.70))["trad"] == "Fragile"
    assert st.verdict(_headline(derisk_sharpe_diff=-0.01, derisk_sharpe=0.49))["trad"] \
        == "Mirage"
    assert st.verdict(_headline(derisk_max_dd=-0.55))["trad"] == "Mirage"


def test_verdict_prose_is_complete_and_honest():
    v = st.verdict(_headline(primary_t=0.9, p_garch=0.08, p_figarch=0.06,
                             derisk_sharpe_diff=-0.01, derisk_sharpe=0.49))
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "mean-reverts" in v["signal_why"]
    assert "does not clear" in v["signal_why"]
    assert "trailed" in v["trad_why"]
    assert "cannot cash" in v["one_sentence"]
