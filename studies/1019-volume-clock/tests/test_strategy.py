"""Strategy tests for Study 1019 — the volume clock, its forecast value, and the verdict rule."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volclock import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes not installed — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_moments_of_a_normal_sample_are_near_zero():
    x = np.random.default_rng(0).normal(size=200_000)
    m = st.moments(x)
    assert abs(m["skew"]) < 0.03 and abs(m["excess_kurtosis"]) < 0.05
    assert m["jb_p"] > 0.001


def test_moments_see_fat_tails():
    x = np.random.default_rng(0).standard_t(5, size=100_000)
    m = st.moments(x)
    assert m["excess_kurtosis"] > 3.0 and m["jb_p"] < 1e-10


def test_moments_decline_on_too_little_data():
    assert np.isnan(st.moments(np.arange(5.0))["excess_kurtosis"])


def test_parkinson_tracks_the_true_variance(planted):
    tape, truth = planted
    pk = st.parkinson_var(tape["high"], tape["low"])
    true_daily = (truth["sigma_ann"] ** 2) / data.TRADING_DAYS
    # a discretely sampled path understates the range a little; level within 25%
    assert pk.mean() == pytest.approx(true_daily, rel=0.25)


def test_relative_volume_is_past_only():
    tape, _ = data.synthetic_tape(n_days=600, seed=1)
    a = st.relative_volume(tape["volume"])
    v2 = tape["volume"].copy()
    v2.iloc[400:] *= 50.0
    b = st.relative_volume(v2)
    assert np.allclose(a.iloc[:400].dropna(), b.iloc[:400].dropna())


def test_relative_volume_removes_the_trend(planted):
    tape, _ = planted
    lr = np.log(st.relative_volume(tape["volume"])).dropna()
    assert abs(lr.iloc[-500:].mean() - lr.iloc[100:600].mean()) < 0.15


def test_garch_fit_recovers_planted_parameters():
    rng = np.random.default_rng(7)
    n, om, a, b = 8000, 2e-6, 0.08, 0.90
    r = np.empty(n)
    v = om / (1 - a - b)
    for t in range(n):
        r[t] = np.sqrt(v) * rng.normal()
        v = om + a * r[t] ** 2 + b * v
    p = st.fit_garch(r)
    assert p["alpha"] == pytest.approx(a, abs=0.03)
    assert p["beta"] == pytest.approx(b, abs=0.04)


def test_garch_forecasts_are_past_only():
    tape, _ = data.synthetic_tape(n_days=1400, seed=3)
    r = st.log_returns(tape["close"])
    f1 = st.garch_forecasts(r, burn=504, refit=252)
    r2 = r.copy()
    r2.iloc[1100:] *= 5.0
    f2 = st.garch_forecasts(r2, burn=504, refit=252)
    assert np.allclose(f1.iloc[:1100].dropna(), f2.iloc[:1100].dropna())
    assert f1.iloc[:504].isna().all() and f1.iloc[504:].notna().all()


# --------------------------------------------------------------------------- #
# Section 1 — the clock removes kurtosis when planted, adds it when not
# --------------------------------------------------------------------------- #
def test_planted_clock_removes_most_of_the_excess_kurtosis(planted):
    tape, _ = planted
    K = st.kurtosis_removed(st.clock_panel(tape), n_boot=200)
    assert K.loc["volume", "share_removed"] > 0.5
    assert K.loc["volume", "share_ci_lo"] > 0.0


def test_unrelated_clock_removes_nothing(null_tape):
    """Dividing by an independent clock adds noise to the denominator: the tails get FATTER."""
    tape, _ = null_tape
    K = st.kurtosis_removed(st.clock_panel(tape), n_boot=200)
    assert K.loc["volume", "share_removed"] < 0.0
    assert K.loc["volume", "share_ci_hi"] < 0.25


def test_share_removed_rises_with_signal_strength():
    shares = []
    for s in (0.0, 0.5, 1.0):
        tape, _ = data.synthetic_tape(n_days=4000, signal_strength=s, seed=1019)
        P = st.clock_panel(tape)
        shares.append(1 - st.moments(P["volume"])["excess_kurtosis"]
                      / st.moments(P["raw"])["excess_kurtosis"])
    assert shares[0] < shares[1] < shares[2]


def test_range_clock_is_an_upper_benchmark_in_both_worlds(planted, null_tape):
    for tape, _ in (planted, null_tape):
        K = st.kurtosis_removed(st.clock_panel(tape), n_boot=50)
        assert K.loc["range", "share_removed"] > 0.9


def test_clock_panel_columns_are_unit_variance_on_a_common_sample(planted):
    tape, _ = planted
    P = st.clock_panel(tape)
    assert list(P.columns) == list(st.CLOCKS)
    assert np.allclose(P.std(ddof=0), 1.0)
    assert P.notna().all().all()


def test_bootstrap_is_deterministic_given_seed(planted):
    tape, _ = planted
    P = st.clock_panel(tape)
    a = st.kurtosis_removed(P, n_boot=50, seed=5)
    b = st.kurtosis_removed(P, n_boot=50, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_tail_table_reads_normal_as_normal():
    z = pd.DataFrame({"raw": np.random.default_rng(0).normal(size=400_000)})
    T = st.tail_table(z)
    assert T.loc["raw", "q1%"] == pytest.approx(-2.326, abs=0.03)
    assert T.loc["raw", "beyond3_vs_normal"] == pytest.approx(1.0, abs=0.1)


def test_exponent_curve_bottoms_near_one_in_clarks_world(planted):
    tape, _ = planted
    E = st.exponent_curve(tape)
    assert 0.6 <= E.idxmin() <= 1.6
    assert E.loc[1.0] < E.loc[0.0]


def test_incremental_removal_is_positive_when_volume_carries_the_variance(planted, null_tape):
    P1 = st.clock_panel(planted[0])
    P0 = st.clock_panel(null_tape[0])
    assert st.incremental_removal(P1, n_boot=100)["ci_lo"] > 0
    assert st.incremental_removal(P0, n_boot=100)["kurt_reduction"] < 0


# --------------------------------------------------------------------------- #
# Section 2 — correlation
# --------------------------------------------------------------------------- #
def test_correlation_detector_fires_on_the_plant_and_not_on_the_null(planted, null_tape):
    C1 = st.volume_vol_correlation(planted[0], lags=(0,))
    C0 = st.volume_vol_correlation(null_tape[0], lags=(0,))
    t1 = C1[C1["target"] == "|r|"]["hac_t"].iloc[0]
    t0 = C0[C0["target"] == "|r|"]["hac_t"].iloc[0]
    assert t1 > 5 and abs(t0) < 2.5


def test_hac_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 0.3 * x + rng.normal(size=5000)
    h = st.hac_slope(y, x)
    assert h["beta"] == pytest.approx(0.3, abs=0.04) and h["t"] > 10


# --------------------------------------------------------------------------- #
# Section 3 — the forecast race
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def races(planted, null_tape):
    return {k: st.score_race(st.forecast_race(t)) for k, (t, _) in
            (("planted", planted), ("null", null_tape))}


def _dm(S, target="pk", loss="qlike", vs="har"):
    return S[(S["target"] == target) & (S["loss"] == loss) & (S["vs"] == vs)].iloc[0]["dm_t"]


def test_lagged_volume_helps_the_forecast_only_when_planted(races):
    assert _dm(races["planted"]) < -2.0
    assert _dm(races["null"]) > -2.0
    assert _dm(races["planted"]) < _dm(races["null"])


def test_forecast_race_is_past_only():
    tape, _ = data.synthetic_tape(n_days=1300, seed=11)
    f1 = st.forecast_race(tape, burn=756)
    t2 = tape.copy()
    t2.iloc[1000:, :] = t2.iloc[1000:, :] * np.array([1.3, 1.6, 1.0, 1.3, 9.0])
    f2 = st.forecast_race(t2, burn=756)
    cols = ["har_r2", "harx_r2", "garch_r2", "har_pk", "harx_pk"]
    # row t forecasts t+1 with data through t: rows strictly before the edit must not move
    a = f1.loc[f1.index < tape.index[999], cols]
    b = f2.loc[f2.index < tape.index[999], cols]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_forecast_race_targets_are_next_day():
    tape, _ = data.synthetic_tape(n_days=1000, seed=2)
    fc = st.forecast_race(tape, burn=756)
    r = st.log_returns(tape["close"])
    t = fc.index[10]
    nxt = tape.index[tape.index.get_loc(t) + 1]
    assert fc.loc[t, "target_r2"] == pytest.approx(r.loc[nxt] ** 2)


def test_qlike_is_minimised_at_the_truth():
    a = np.full(1000, 2.0)
    assert st.qlike(a, 2.0).mean() < st.qlike(a, 1.5).mean()
    assert st.qlike(a, 2.0).mean() < st.qlike(a, 3.0).mean()
    assert np.isfinite(st.qlike(np.zeros(3), 1.0)).all()   # a flat day is not a crash


def test_diebold_mariano_signs():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.2, 3000)
    worse = base + 0.05 + rng.normal(0, 0.05, 3000)
    assert st.diebold_mariano(worse, base)["dm"] > 5
    assert st.diebold_mariano(base, worse)["dm"] < -5
    assert np.isnan(st.diebold_mariano(base[:10], base[:10])["dm"])


# --------------------------------------------------------------------------- #
# Section 4 — overlay plumbing
# --------------------------------------------------------------------------- #
def _toy_overlay(cost=0.0):
    idx = pd.bdate_range("2001-01-01", periods=300)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=idx)
    var = pd.Series(np.linspace(1e-4, 4e-4, 300), index=idx)
    rf = pd.Series(0.0, index=idx)
    return st.overlay(r, var, rf, cost_bps=cost), var


def test_overlay_has_exactly_one_execution_lag():
    """Forecast at close t, trade at close t+1, earn t+2 — never the t+1 return."""
    ov, var = _toy_overlay()
    w = np.minimum(1.5, 0.15 / np.sqrt(252) / np.sqrt(var))
    assert st.EXECUTION_LAG == 1
    lagged = w.shift(2).reindex(ov.index)
    assert np.allclose(ov["pos"].to_numpy(), lagged.to_numpy())
    assert ov.index[0] > var.index[1]


def test_costs_only_ever_reduce_the_overlay():
    g, _ = _toy_overlay(0.0)
    n, _ = _toy_overlay(10.0)
    assert (n["net"] <= g["net"] + 1e-15).all()
    assert n["net"].sum() < g["net"].sum()


def test_sharpe_diff_of_a_series_with_itself_is_zero():
    x = pd.Series(np.random.default_rng(1).normal(0.0003, 0.01, 2000),
                  index=pd.bdate_range("2001-01-01", periods=2000))
    t = st.sharpe_diff_test(x, x, n_boot=100)
    assert t["diff"] == 0.0 and t["p"] == 1.0


def test_sharpe_diff_detects_a_clearly_better_series():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2001-01-01", periods=5000)
    common = rng.normal(0, 0.01, 5000)
    a = pd.Series(common + 0.001, index=idx)
    b = pd.Series(common, index=idx)
    t = st.sharpe_diff_test(a, b, n_boot=300)
    assert t["diff"] > 0 and t["p"] < 0.05


def test_overlay_race_runs_and_scores_every_model(planted):
    tape, _ = planted
    fc = st.forecast_race(tape)
    rf = pd.Series(0.0001, index=tape.index)
    O = st.overlay_race(tape, fc, rf, costs_bps=(0.0, 5.0), n_boot=100)
    assert set(O["table"]["model"]) == {"har", "harx", "garch", "buy & hold"}
    assert len(O["tests"]) == 4


# --------------------------------------------------------------------------- #
# The verdict rule — both directions on every axis
# --------------------------------------------------------------------------- #
def _tape(**over):
    t = {"k_raw": 9.0, "k_volume": 3.0, "share_removed": 0.66, "share_ci_lo": 0.4,
         "share_ci_hi": 0.8, "share_removed_range": 1.05, "share_removed_garch": 0.75,
         "share_removed_garch_volume": 0.85, "jb_p_volume": 0.0, "corr0": 0.25,
         "corr0_t": 8.0, "corr1": 0.1, "corr1_t": 3.0, "dm_t": -3.0, "dm_p": 0.003,
         "dm_t_pk": -2.5, "dm_t_vs_garch": -5.0, "sharpe_diff_net": 0.05,
         "sharpe_diff_p": 0.01, "lookahead_sharpe": 0.7, "lookahead_base_sharpe": 0.5,
         "har_sharpe": 0.4, "inc_ci_lo": 0.2}
    t.update(over)
    return t


def _h(a=None, b=None):
    return {"tapes": {"sp500": _tape(**(a or {})), "nasdaq": _tape(**(b or {}))},
            "labels": dict(data.TAPE_LABELS), "base_cost_bps": 2.0}


def test_verdict_signal_real_requires_every_tape():
    assert st.verdict(_h())["signal"] == "Real"
    assert st.verdict(_h(a={"share_removed": 0.14, "share_ci_lo": -0.1}))["signal"] == "Weak"


def test_verdict_signal_needs_a_robust_correlation_not_just_kurtosis():
    assert st.verdict(_h(a={"corr0_t": 1.5}, b={"corr0_t": 1.5}))["signal"] == "Weak"


def test_verdict_signal_mixed_and_none():
    assert st.verdict(_h(b={"share_removed": -0.2, "share_ci_lo": -0.5}))["signal"] == "Mixed"
    neg = {"share_removed": -0.2, "share_ci_lo": -0.5}
    assert st.verdict(_h(a=neg, b=neg))["signal"] == "None"


def test_verdict_tradability_investable_only_when_everything_survives():
    assert st.verdict(_h())["trad"] == "Investable"
    assert st.verdict(_h(a={"sharpe_diff_p": 0.3}))["trad"] == "Fragile"


def test_verdict_tradability_fragile_and_mirage():
    no_fc = {"dm_t": 0.5, "dm_p": 0.6}
    assert st.verdict(_h(a=no_fc))["trad"] == "Fragile"
    assert st.verdict(_h(a=no_fc, b=no_fc))["trad"] == "Mirage"
    lose = {"sharpe_diff_net": -0.01, "sharpe_diff_p": 0.5}
    assert st.verdict(_h(a=lose, b=lose))["trad"] == "Mirage"


def test_verdict_contract_and_prose():
    v = st.verdict(_h(a={"share_removed": 0.14, "share_ci_lo": -0.1},
                      b={"dm_t": 0.1, "dm_p": 0.9}))
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "Jarque–Bera" in v["signal_why"]
    assert "wrong side of the close" in v["trad_why"]
    assert "S&P 500" in v["signal_why"] and "Nasdaq" in v["trad_why"]


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_returns_are_fat_tailed_and_volume_tracks_volatility_same_day():
    for name in data.TAPES:
        tape = data.load_tape(name)
        assert st.moments(st.log_returns(tape["close"]))["excess_kurtosis"] > 3
        C = st.volume_vol_correlation(tape, lags=(0, 1))
        same = C[(C["lag"] == 0) & (C["target"] == "|r|")].iloc[0]
        nxt = C[(C["lag"] == 1) & (C["target"] == "|r|")].iloc[0]
        assert same["hac_t"] > 2 and same["corr"] > nxt["corr"]


@needs_real
def test_real_clock_ordering_range_beats_garch_beats_volume():
    tape = data.load_tape("sp500")
    P = st.clock_panel(tape)
    k = {c: st.moments(P[c])["excess_kurtosis"] for c in P.columns}
    assert abs(k["range"]) < k["garch"] < k["volume"] < k["raw"]
