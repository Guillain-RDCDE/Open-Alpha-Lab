"""Strategy tests for Study 1027 — the estimators must find a planted mirror, a planted
equity effect, and nothing on the symmetric null; the overlay must be lagged once; the verdict
rule must move in both directions."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mirrorlev import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes not installed — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Parametric: GJR and EGARCH recover sign and size
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["gjr", "egarch"])
def test_parametric_fires_on_the_planted_mirror(mirror_world, model):
    f = st.fit_asymmetry(mirror_world[1], model)
    assert f["converged"]
    assert f["inv_t"] > 2.0


@pytest.mark.parametrize("model", ["gjr", "egarch"])
def test_parametric_reads_the_equity_sign_as_negative(equity_world, model):
    assert st.fit_asymmetry(equity_world[1], model)["inv_t"] < -2.0


@pytest.mark.parametrize("model", ["gjr", "egarch"])
def test_parametric_stays_quiet_on_the_null(null_world, model):
    assert abs(st.fit_asymmetry(null_world[1], model)["inv_t"]) < 2.0


def test_gjr_recovers_the_planted_gamma(mirror_world, equity_world):
    for _, r, truth in (mirror_world, equity_world):
        f = st.fit_asymmetry(r, "gjr")
        assert f["gamma"] == pytest.approx(truth["gamma"], abs=0.04)


def test_gjr_down_up_ratio_points_the_right_way(mirror_world, equity_world):
    assert st.fit_asymmetry(equity_world[1], "gjr")["down_up_ratio"] > 1.5
    assert st.fit_asymmetry(mirror_world[1], "gjr")["down_up_ratio"] < 0.7


def test_fit_declines_on_too_little_data():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    assert st.fit_asymmetry(r) == {}


# --------------------------------------------------------------------------- #
# Model-free
# --------------------------------------------------------------------------- #
def test_forward_vol_is_strictly_future():
    r = pd.Series(np.zeros(60), index=pd.bdate_range("2000-01-03", periods=60))
    r.iloc[30] = 0.10
    fwd = st.forward_vol(r, 5)
    # the shock on day 30 is in the forward window of days 25..29, never of day 30 itself
    assert fwd.iloc[30] == 0.0
    assert fwd.iloc[29] > 0 and fwd.iloc[25] > 0 and fwd.iloc[24] == 0.0


def test_model_free_fires_on_the_mirror_and_on_equity_with_opposite_signs(mirror_world,
                                                                         equity_world):
    assert st.forward_vol_regression(mirror_world[1], 21)["diff_t"] > 2.0
    assert st.forward_vol_regression(equity_world[1], 21)["diff_t"] < -2.0


def test_model_free_quiet_on_null(null_world):
    assert abs(st.forward_vol_regression(null_world[1], 21)["diff_t"]) < 2.0


def test_sign_correlation_and_matched_ratio(mirror_world, null_world):
    m = st.sign_correlation(mirror_world[1], 21, n_boot=150)
    n = st.sign_correlation(null_world[1], 21, n_boot=150)
    assert m["corr"] > 0 and m["lo"] > 0 and m["matched_up_over_down"] > 1.0
    assert n["lo"] < 0 < n["hi"]


def test_block_bootstrap_indices_are_circular_and_complete():
    idx = st._block_indices(100, 21, np.random.default_rng(0))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def _spliced():
    """First half mirror, second half equity: a regime the era machinery must see."""
    a, _ = data.synthetic_gjr(n_years=10, signal_strength=-1.0, seed=7, start="2000-01-03")
    b, _ = data.synthetic_gjr(n_years=10, signal_strength=1.0, seed=8, start="2010-01-01")
    return pd.concat([data.log_returns(a), data.log_returns(b)])


def test_era_table_splits_a_planted_regime():
    r = _spliced()
    eras = {"A": ("2000-01-01", "2009-12-31"), "B": ("2010-01-01", "2020-12-31")}
    tbl = st.era_table(r, eras)
    assert tbl.loc["A", "gjr_inv_t"] > 2 and tbl.loc["B", "gjr_inv_t"] < -2
    d = st.era_difference(tbl, "A", "B")
    assert d["z"] > 2


def test_rolling_asymmetry_tracks_the_regime():
    roll = st.rolling_asymmetry(_spliced(), window=756, step=252)
    early = roll.loc[:"2009-06-30", "inv"].mean()
    late = roll.loc["2013-01-01":, "inv"].mean()
    assert early > 0 > late


# --------------------------------------------------------------------------- #
# Skewness
# --------------------------------------------------------------------------- #
def test_skew_ci_brackets_a_known_skew():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.exponential(1.0, 4000) - 1.0)        # skew = 2
    s = st.skew_ci(x, n_boot=200)
    assert s["lo"] < 2.0 < s["hi"] + 0.3 and s["quantile_skew"] > 0
    y = pd.Series(rng.normal(size=4000))
    assert abs(st.skew_ci(y, n_boot=100)["skew"]) < 0.2


def test_skew_difference_is_zero_against_itself():
    x = pd.Series(np.random.default_rng(1).normal(size=2000),
                  index=pd.bdate_range("2000-01-03", periods=2000))
    d = st.skew_difference(x, x.rename("y"), n_boot=50)
    assert d["diff"] == pytest.approx(0.0) and d["p"] == 1.0


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #
def test_overlay_uses_exactly_one_lag():
    """A return on day t must earn the weight decided at t-1, not at t."""
    px, _ = data.synthetic_gjr(n_years=3, seed=3)
    r = data.simple_returns(px)
    o = st.vol_target_overlay(r, None, cost_bps=0.0, n_boot=2)
    w_dec = st.vol_target_weights(np.log1p(r))
    s = o["series"]
    expected = (w_dec.shift(1) * r).reindex(s.index)
    assert np.allclose(s["gross"].to_numpy(), expected.to_numpy())


def test_overlay_costs_are_one_way_times_turnover():
    px, _ = data.synthetic_gjr(n_years=3, seed=3)
    r = data.simple_returns(px)
    a = st.vol_target_overlay(r, None, cost_bps=0.0, n_boot=2)["series"]
    b = st.vol_target_overlay(r, None, cost_bps=10.0, n_boot=2)["series"]
    assert np.allclose((a["net"] - b["net"]).to_numpy(), 10.0 / 1e4 * b["turn"].to_numpy())


def test_overlay_subtracts_rf_on_both_legs():
    px, _ = data.synthetic_gjr(n_years=3, seed=3)
    r = data.simple_returns(px)
    rf = pd.Series(0.0001, index=r.index)
    o = st.vol_target_overlay(r, rf, cost_bps=0.0, n_boot=2)["series"]
    assert np.allclose(o["bh"].to_numpy(), (r - rf).reindex(o.index).to_numpy())


def test_thermostat_reacts_to_the_sign_the_world_plants(mirror_world, equity_world, null_world):
    """On the equity world it de-risks after falls (reacts > 0); on the mirror, after rallies."""
    react = {}
    for lab, w in (("eq", equity_world), ("mir", mirror_world), ("null", null_world)):
        react[lab] = st.vol_target_overlay(data.simple_returns(w[0]), None,
                                           n_boot=2)["react_corr"]
    assert react["eq"] > 0.05 > -0.05 > react["mir"]
    assert react["eq"] > react["null"] > react["mir"]


def test_weights_are_capped_and_past_only():
    px, _ = data.synthetic_gjr(n_years=3, seed=4)
    r = data.log_returns(px)
    w = st.vol_target_weights(r, cap=2.0)
    assert w.max() <= 2.0 and w.iloc[:251].isna().all()
    r2 = r.copy()
    r2.iloc[-1] = 0.5                                  # change only the last day
    w2 = st.vol_target_weights(r2, cap=2.0)
    assert np.allclose(w.iloc[:-1].to_numpy(), w2.iloc[:-1].to_numpy(), equal_nan=True)


def test_drawdown_and_sharpe_helpers():
    assert st.max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    x = np.random.default_rng(0).normal(0.001, 0.01, 5000)
    assert st.sharpe(x) == pytest.approx(0.001 / 0.01 * np.sqrt(252), rel=0.15)


def test_cost_sweep_is_monotone():
    px, _ = data.synthetic_gjr(n_years=4, seed=5)
    cs = st.cost_sweep(data.simple_returns(px), None, costs=(0.0, 10.0, 50.0))
    assert cs["gain_net"].is_monotonic_decreasing


def test_monthly_asymmetry_runs():
    px, _ = data.synthetic_gjr(n_years=30, signal_strength=-1.0, seed=6)
    m = st.monthly_asymmetry(px.resample("ME").last())
    assert np.isfinite(m["diff_t"]) and m["n"] > 300


# --------------------------------------------------------------------------- #
# The verdict rule — both directions
# --------------------------------------------------------------------------- #
def _h(**over):
    h = {"wti_gjr_inv_t": 2.6, "wti_egarch_inv_t": 3.1, "wti_mf_t": 2.4, "wti_mf5_t": 1.9,
         "sp_gjr_inv_t": -8.0, "era_labels": ["1986-2007", "2008-2018"],
         "era_inv_t": [2.5, 1.2], "era_inv": [0.05, 0.02], "era_diff_z": 1.0,
         "brent_mf_t": 0.8, "ov_wti_gain_net": 0.05, "ov_wti_gain_lo": -0.05,
         "ov_wti_gain_hi": 0.15, "ov_wti_mdd_bh": -0.80, "ov_wti_mdd_net": -0.60,
         "ov_wti_sharpe_bh": 0.25, "ov_wti_sharpe_net": 0.30, "ov_wti_cost_bps": 10.0,
         "ov_wti_react": -0.2, "ov_sp_gain_net": 0.1, "ov_sp_gain_lo": -0.1,
         "ov_sp_gain_hi": 0.3, "ov_sp_react": 0.3}
    h.update(over)
    return h


def test_verdict_real_needs_both_lenses_at_the_bar():
    assert st.verdict(_h())["signal"] == "Real"
    assert st.verdict(_h(wti_mf_t=1.5))["signal"] == "Weak"
    assert st.verdict(_h(wti_gjr_inv_t=1.5))["signal"] == "Weak"


def test_verdict_none_when_neither_lens_sees_the_mirror():
    h = _h(wti_gjr_inv_t=-2.6, wti_mf_t=-1.3, era_inv_t=[-0.2, -4.8], era_inv=[-0.002, -0.06],
           era_diff_z=-3.4)
    v = st.verdict(h)
    assert v["signal"] == "None"
    assert "fainter copy" in v["one_sentence"]


def test_verdict_mixed_when_eras_split_both_ways():
    h = _h(wti_gjr_inv_t=0.5, wti_mf_t=0.4, era_inv_t=[3.0, -3.0], era_inv=[0.05, -0.05])
    assert st.verdict(h)["signal"] == "Mixed"
    h2 = _h(wti_gjr_inv_t=0.5, wti_mf_t=0.4, era_inv_t=[3.0, -1.0], era_inv=[0.05, -0.01],
            era_diff_z=2.5)
    assert st.verdict(h2)["signal"] == "Mixed"
    h3 = _h(wti_gjr_inv_t=0.5, wti_mf_t=0.4, era_inv_t=[3.0, 1.0], era_inv=[0.05, 0.01],
            era_diff_z=2.5)
    assert st.verdict(h3)["signal"] == "Weak"


def test_verdict_tradability_never_investable_on_a_spot_tape():
    assert st.verdict(_h())["trad"] == "Fragile"
    assert st.verdict(_h(ov_wti_gain_net=-0.08))["trad"] == "Mirage"
    assert st.verdict(_h(ov_wti_mdd_net=-0.86))["trad"] == "Mirage"
    assert st.verdict(_h(ov_wti_gain_net=5.0))["trad"] != "Investable"


def test_verdict_shape_and_prose():
    v = st.verdict(_h())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "spot" in v["trad_why"] and "Positive = volatility rises after rallies" in v["signal_why"]


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_sp500_shows_the_textbook_leverage_effect():
    r = data.log_returns(data.load_sp500())
    assert st.fit_asymmetry(r, "gjr")["inv_t"] < -2.0


@needs_real
def test_real_wti_does_not_show_the_mirror():
    r = data.log_returns(data.load_wti())
    assert st.fit_asymmetry(r, "gjr")["inv_t"] < 2.0
    assert st.forward_vol_regression(r, 21)["diff_t"] < 2.0
