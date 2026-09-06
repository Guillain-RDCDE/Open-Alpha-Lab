"""Strategy tests for Study 988 — is the decay real, or is it the start date?"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from taming import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The calendar
# --------------------------------------------------------------------------- #
def test_a_daily_asset_annualises_on_365():
    px = st.synthetic_world(n=2000)
    assert st.annualisation_factor(px) == pytest.approx(365, abs=3)


def test_a_weekday_asset_annualises_on_252():
    idx = pd.bdate_range("2015-01-01", periods=2000)
    px = pd.Series(np.linspace(100, 200, 2000), index=idx)
    assert st.annualisation_factor(px) == pytest.approx(261, abs=6)


def test_using_the_wrong_calendar_is_a_20_percent_error():
    """The reason this study infers the factor rather than hard-coding it."""
    px = st.synthetic_world(n=2000, decay_per_year=0.0)
    right = st.realised_vol(px, 30, st.CRYPTO_DAYS).mean()
    wrong = st.realised_vol(px, 30, st.EQUITY_DAYS).mean()
    assert right / wrong == pytest.approx(np.sqrt(365 / 252), rel=0.01)
    assert right / wrong > 1.15


def test_annualisation_factor_falls_back_on_a_short_series():
    s = pd.Series([1.0] * 5, index=pd.bdate_range("2020-01-01", periods=5))
    assert st.annualisation_factor(s) == st.EQUITY_DAYS


def test_realised_vol_recovers_a_planted_level():
    rng = np.random.default_rng(988)
    n = 4000
    r = rng.normal(0, 0.80 / np.sqrt(365), n)
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.date_range("2015-01-01", periods=n))
    assert st.realised_vol(px, 365).mean() == pytest.approx(0.80, rel=0.1)


def test_vol_summary_shows_the_persistence():
    px = st.synthetic_world(n=4000, persistence=0.995)
    s = st.vol_summary(px, windows=(30, 90))
    assert s.loc[30, "autocorr_100"] > 0.1
    assert s.loc[90, "mean"] > 0


# --------------------------------------------------------------------------- #
# The four estimators
# --------------------------------------------------------------------------- #
def test_ols_recovers_a_planted_decay():
    px = st.synthetic_world(n=6000, decay_per_year=-0.15, persistence=0.98)
    v = st.realised_vol(px, 90)
    o = st.ols_trend(v)
    assert o["slope_per_year"] == pytest.approx(-0.15, abs=0.08)


def test_ols_finds_no_trend_when_none_is_planted_on_average():
    slopes = []
    for s in range(10):
        px = st.synthetic_world(n=4000, decay_per_year=0.0, persistence=0.98, seed=988 + s)
        slopes.append(st.ols_trend(st.realised_vol(px, 90))["slope_per_year"])
    assert abs(np.mean(slopes)) < 0.10


def test_but_a_single_null_run_can_easily_look_like_a_trend():
    """The study's central failure mode, as a test: persistence manufactures trends."""
    slopes = []
    for s in range(20):
        px = st.synthetic_world(n=4000, decay_per_year=0.0, persistence=0.995, seed=988 + s)
        slopes.append(st.ols_trend(st.realised_vol(px, 90))["t_naive"])
    slopes = np.array(slopes)
    assert (np.abs(slopes) > 2).mean() > 0.5      # a naive t rejects far too often


def test_theil_sen_agrees_with_ols_on_a_clean_trend():
    px = st.synthetic_world(n=6000, decay_per_year=-0.15, persistence=0.95)
    v = st.realised_vol(px, 90)
    assert st.theil_sen(v)["slope_per_year"] == pytest.approx(
        st.ols_trend(v)["slope_per_year"], abs=0.10)


def test_theil_sen_is_less_moved_by_a_single_spike():
    px = st.synthetic_world(n=4000, persistence=0.95)
    v = st.realised_vol(px, 90)
    spiked = v.copy()
    spiked.iloc[100:130] *= 8
    d_ols = abs(st.ols_trend(spiked)["slope_per_year"] - st.ols_trend(v)["slope_per_year"])
    d_ts = abs(st.theil_sen(spiked)["slope_per_year"] - st.theil_sen(v)["slope_per_year"])
    assert d_ts < d_ols


def test_the_block_bootstrap_standard_error_is_much_wider_than_the_naive_one():
    """The single most important line in the module."""
    px = st.synthetic_world(n=5000, decay_per_year=-0.10, persistence=0.995)
    v = st.realised_vol(px, 90)
    o = st.ols_trend(v)
    bb = st.block_bootstrap_trend(v, n_boot=300)
    assert bb["boot_sd"] > o["se_naive"] * 2
    assert abs(bb["t_boot"]) < abs(o["t_naive"])


def test_block_bootstrap_keeps_the_point_estimate():
    px = st.synthetic_world(n=5000, decay_per_year=-0.10)
    v = st.realised_vol(px, 90)
    assert st.block_bootstrap_trend(v, n_boot=200)["slope_per_year"] == pytest.approx(
        st.ols_trend(v)["slope_per_year"], abs=1e-9)


def test_mann_kendall_finds_the_direction():
    down = st.realised_vol(st.synthetic_world(n=6000, decay_per_year=-0.20,
                                              persistence=0.95), 90)
    up = st.realised_vol(st.synthetic_world(n=6000, decay_per_year=0.20,
                                            persistence=0.95), 90)
    assert st.mann_kendall(down)["direction"] == "down"
    assert st.mann_kendall(up)["direction"] == "up"


def test_mann_kendall_thins_the_sample():
    v = st.realised_vol(st.synthetic_world(n=6000), 90)
    mk = st.mann_kendall(v, sample=500)
    assert mk["n"] < mk["thinned_from"]
    assert mk["n"] <= 600


def test_every_estimator_declines_on_too_little_data():
    v = pd.Series([0.5] * 40, index=pd.date_range("2020-01-01", periods=40))
    for fn in (st.ols_trend, st.theil_sen, st.mann_kendall, st.block_bootstrap_trend):
        assert "slope_per_year" not in fn(v) or np.isnan(fn(v).get("slope_per_year", np.nan))


def test_trend_table_reports_all_four():
    v = st.realised_vol(st.synthetic_world(n=4000, decay_per_year=-0.1), 90)
    t = st.trend_table(v)
    assert len(t) == 4
    assert "Mann-Kendall (rank)" in t.index


# --------------------------------------------------------------------------- #
# The start-date control
# --------------------------------------------------------------------------- #
def test_start_date_sensitivity_refits_from_many_starts():
    v = st.realised_vol(st.synthetic_world(n=6000), 90)
    s = st.start_date_sensitivity(v, step=90)
    assert len(s) > 10
    assert s["years"].is_monotonic_decreasing


def test_a_genuine_decay_survives_almost_every_start_date():
    v = st.realised_vol(st.synthetic_world(n=8000, decay_per_year=-0.30,
                                           persistence=0.95), 90)
    s = st.sensitivity_summary(st.start_date_sensitivity(v, step=90))
    assert s["share_negative"] > 0.9


def test_a_null_world_gives_start_dates_that_disagree_with_each_other():
    """If the answer depends on where you start, there is no answer."""
    v = st.realised_vol(st.synthetic_world(n=8000, decay_per_year=0.0,
                                           persistence=0.995), 90)
    s = st.sensitivity_summary(st.start_date_sensitivity(v, step=90))
    assert 0.0 < s["share_negative"] < 1.0
    assert s["max_slope"] > 0 > s["min_slope"]


def test_the_fitted_slope_is_driven_by_the_starting_volatility():
    """The trick behind every 'maturing' chart, measured."""
    v = st.realised_vol(st.synthetic_world(n=8000, decay_per_year=0.0,
                                           persistence=0.995), 90)
    s = st.sensitivity_summary(st.start_date_sensitivity(v, step=60))
    assert s["corr_with_start_vol"] < -0.2


def test_sensitivity_summary_handles_an_empty_frame():
    assert st.sensitivity_summary(pd.DataFrame())["n"] == 0


def test_start_date_sensitivity_is_empty_on_a_short_series():
    v = st.realised_vol(st.synthetic_world(n=500), 90)
    assert st.start_date_sensitivity(v, min_years=3.0).empty


# --------------------------------------------------------------------------- #
# Cuts
# --------------------------------------------------------------------------- #
def test_by_era_splits_the_sample_evenly():
    v = st.realised_vol(st.synthetic_world(n=4000), 90)
    e = st.by_era(v, n_eras=4)
    assert len(e) == 4
    assert abs(e["n"].max() - e["n"].min()) <= 1


def test_halving_alignment_needs_data_around_each_event():
    v = st.realised_vol(st.synthetic_world(n=4000), 90)
    out = st.halving_alignment(v, halvings=("2050-01-01",))
    assert out.empty


def test_relative_to_equities_is_a_ratio():
    c = st.synthetic_world(n=3000, base_vol=0.80)
    e = st.synthetic_world(n=3000, base_vol=0.16, seed=1)
    r = st.relative_to_equities(c, {"equity": e}, window=365)
    assert r["equity"].median() > 2.0


# --------------------------------------------------------------------------- #
# The sizing rule
# --------------------------------------------------------------------------- #
def test_vol_targeting_delivers_roughly_its_target():
    px = st.synthetic_world(n=6000, base_vol=0.80, persistence=0.98)
    out = st.sizing_backtest(px, target_vol=0.40, cost_bps=0.0, max_leverage=10.0)
    assert out["vol_targeted"]["vol"] == pytest.approx(0.40, rel=0.35)


def test_vol_targeting_reduces_volatility_versus_holding():
    px = st.synthetic_world(n=6000, base_vol=0.80)
    out = st.sizing_backtest(px, target_vol=0.30, cost_bps=0.0)
    assert out["vol_targeted"]["vol"] < out["buy_hold"]["vol"]


def test_the_leverage_cap_binds():
    px = st.synthetic_world(n=3000, base_vol=0.20)
    out = st.sizing_backtest(px, target_vol=2.0, cost_bps=0.0, max_leverage=1.5)
    assert out["mean_leverage"] <= 1.5


def test_costs_reduce_the_vol_targeted_return():
    px = st.synthetic_world(n=4000)
    free = st.sizing_backtest(px, cost_bps=0.0)
    paid = st.sizing_backtest(px, cost_bps=50.0)
    assert paid["vol_targeted"]["cagr"] < free["vol_targeted"]["cagr"]


def test_a_decaying_world_forces_a_vol_targeter_to_lever_up():
    """The practical stake in the trend question."""
    px = st.synthetic_world(n=8000, decay_per_year=-0.25, persistence=0.95)
    out = st.sizing_backtest(px, target_vol=0.40, cost_bps=0.0, max_leverage=20.0)
    assert out["leverage_trend"] > 0.05


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"window": 30, "mean_vol": 0.62, "min_vol": 0.17, "max_vol": 1.85, "years": 11.8,
         "ols_pct": -0.041, "ols_t": -9.4, "ts_pct": -0.036, "boot_slope": -0.042,
         "boot_t": -1.6, "autocorr_100": 0.31, "share_negative": 0.74,
         "share_significant_down": 0.52, "corr_with_start_vol": -0.68,
         "target_vol": 0.40, "vt_cagr": 0.41, "bh_cagr": 0.55, "vt_sharpe": 0.72,
         "bh_sharpe": 0.66, "vt_dd": -0.51, "bh_dd": -0.77, "mean_leverage": 0.81,
         "leverage_trend": 0.03}
    h.update(over)
    return h


def test_verdict_signal_needs_significance_and_start_date_robustness():
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(boot_t=-2.5))["signal"] == "Weak"          # still not robust
    assert st.verdict(_headline(boot_t=-2.5,
                                share_significant_down=0.9))["signal"] == "Real"
    assert st.verdict(_headline(boot_slope=0.01))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Partial"
    assert st.verdict(_headline(vt_sharpe=0.90))["trad"] == "Useful"
    assert st.verdict(_headline(vt_sharpe=0.40))["trad"] == "Mirage"


def test_verdict_prose_names_the_start_date_trick():
    v = st.verdict(_headline())
    assert "start date" in v["signal_why"] or "start at a peak" in v["signal_why"]
    assert "start" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
