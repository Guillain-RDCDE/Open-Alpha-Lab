"""Strategy tests for Study 1020 — is the market more volatile than ever?

Synthetic-first: every detector is shown to fire on a planted trend and stay quiet on the
matched null. The real-tape checks at the bottom are gated on the bundled tapes.
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volhistory import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Measuring volatility
# --------------------------------------------------------------------------- #
def test_annual_rv_keeps_only_complete_years():
    idx = pd.date_range("1926-07-31", "1929-11-30", freq="ME")
    r = pd.Series(0.01, index=idx)
    rv = st.annual_rv_from_monthly(r)
    assert list(rv.index) == [1927, 1928]
    assert rv.iloc[0] == pytest.approx(np.sqrt(12) * np.log1p(0.01))


def test_annual_rv_from_daily_recovers_a_constant_volatility():
    f, _ = data.synthetic_returns(n_years=10, freq="D", signal_strength=0.0, logvol_sd=1e-9,
                                  seed=3)
    rv = st.annual_rv_from_daily(f["price"])
    assert rv.mean() == pytest.approx(0.16, rel=0.05)


def test_parkinson_matches_its_formula():
    idx = pd.bdate_range("2000-01-03", periods=260)
    o = pd.DataFrame({"open": 100.0, "close": 100.0, "high": 101.0, "low": 99.0}, index=idx)
    v = st.parkinson_daily_var(o)
    assert v.iloc[0] == pytest.approx(np.log(101 / 99) ** 2 / (4 * np.log(2)))
    assert st.annual_rv_parkinson(o).iloc[0] == pytest.approx(np.sqrt(252 * v.iloc[0]))


def test_monthly_rv_is_annualised_variance():
    f, _ = data.synthetic_returns(n_years=5, freq="D", signal_strength=0.0, logvol_sd=1e-9,
                                  seed=4)
    mv = st.monthly_rv_from_daily(f["price"])
    assert np.sqrt(mv.mean()) == pytest.approx(0.16, rel=0.08)


# --------------------------------------------------------------------------- #
# Trend inference
# --------------------------------------------------------------------------- #
def test_slope_stats_recover_an_exact_line():
    y = 0.5 + 0.02 * np.arange(50) + np.random.default_rng(0).normal(0, 1e-6, 50)
    b, t_ols, t_nw, t_kvb = st._slope_stats(y[None, :], lags=3)
    assert b[0] == pytest.approx(0.02, abs=1e-6)
    assert t_ols[0] > 100 and t_nw[0] > 100 and t_kvb[0] > 100


def test_kvb_critical_value_is_fat_tailed_and_pvalue_monotone():
    c = st.kvb_critical(0.05)
    assert 3.5 < c < 9.0          # far beyond 1.96: the fixed-b limit is not normal
    assert st.kvb_pvalue(0.0) == pytest.approx(1.0)
    assert st.kvb_pvalue(1.0) > st.kvb_pvalue(c) > st.kvb_pvalue(3 * c)
    assert st.kvb_pvalue(c) == pytest.approx(0.05, abs=0.01)


def test_kvb_has_the_right_size_on_iid_noise():
    rng = np.random.default_rng(99)
    Y = rng.standard_normal((800, 60))
    _, _, _, t = st._slope_stats(Y, lags=0)
    rej = np.mean([st.kvb_pvalue(x) < 0.05 for x in t])
    assert 0.02 < rej < 0.09


def test_trend_test_fires_on_the_plant(planted):
    f, truth = planted
    tr = st.trend_test(np.log(st.annual_rv_from_monthly(f["ret"])), n_boot=499)
    assert tr["slope"] == pytest.approx(truth["trend_log_per_year"], abs=0.006)
    assert st.passes(tr)


def test_trend_test_stays_quiet_on_the_null(null_world):
    f, _ = null_world
    tr = st.trend_test(np.log(st.annual_rv_from_monthly(f["ret"])), n_boot=499)
    assert not st.passes(tr)
    assert not st.passes(tr, direction=-1)
    assert tr["p_kvb"] > 0.05


def test_detector_power_and_size_across_worlds():
    """The machinery proof: fires on most planted worlds, rarely on the null."""
    hit = st.synthetic_detection(1.0, seeds=range(12))
    miss = st.synthetic_detection(0.0, seeds=range(12))
    assert hit["fire_rate"] >= 0.6
    assert miss["fire_rate"] <= 0.17
    assert hit["mean_slope"] == pytest.approx(hit["true_slope"], abs=0.004)


def test_the_robust_bar_is_more_honest_than_naive_ols_on_a_persistent_null():
    """On a very persistent stationary volatility, naive OLS finds trends that are not there."""
    r = st.synthetic_detection(0.0, seeds=range(25), monthly_phi=0.99)
    assert r["fire_rate"] <= r["ols_rate"]
    assert r["fire_rate"] <= 0.12


def test_passes_respects_direction():
    tr = {"slope": 0.01, "t_nw": 2.5, "p_kvb": 0.01}
    assert st.passes(tr) and not st.passes(tr, direction=-1)
    assert not st.passes({"slope": 0.01, "t_nw": 2.5, "p_kvb": 0.2})
    assert not st.passes({"slope": 0.01, "t_nw": 1.5, "p_kvb": 0.01})
    assert not st.passes({})


def test_trend_from_every_start_covers_every_start(null_world):
    f, _ = null_world
    y = np.log(st.annual_rv_from_monthly(f["ret"]))
    s = st.trend_from_every_start(y, min_len=20)
    assert len(s) == len(y) - 20 + 1
    assert s["n"].iloc[0] == len(y) and s["n"].iloc[-1] == 20


# --------------------------------------------------------------------------- #
# Eras and decades
# --------------------------------------------------------------------------- #
def test_decade_contrast_detects_a_doubling():
    rng = np.random.default_rng(5)
    idx = pd.date_range("1930-01-31", "1949-12-31", freq="ME")
    sd = np.where(idx.year < 1940, 0.08, 0.04)
    r = pd.Series(rng.normal(0, sd), index=idx)
    dc = st.decade_contrast(r, (1930, 1939), (1940, 1949), n_boot=1000)
    assert dc["ratio"] == pytest.approx(2.0, rel=0.25)
    assert dc["ci_lo"] > 0 and dc["p"] < 0.01


def test_decade_contrast_is_quiet_when_nothing_changed():
    rng = np.random.default_rng(6)
    idx = pd.date_range("1930-01-31", "1949-12-31", freq="ME")
    r = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx)
    dc = st.decade_contrast(r, (1930, 1939), (1940, 1949), n_boot=1000)
    assert dc["ci_lo"] < 0 < dc["ci_hi"]


def test_era_table_has_one_row_per_era(null_world):
    f, _ = null_world
    e = st.era_table(f["ret"], data.ERAS, n_boot=200)
    assert list(e.index) == [x[0] for x in data.ERAS[:len(e)]]
    assert (e["ci_lo"] <= e["vol"]).all() and (e["vol"] <= e["ci_hi"]).all()


# --------------------------------------------------------------------------- #
# Extreme days
# --------------------------------------------------------------------------- #
def test_poisson_ci_matches_known_values():
    assert st.poisson_ci(0) == pytest.approx((0.0, 3.689), abs=1e-3)
    lo, hi = st.poisson_ci(10)
    assert lo == pytest.approx(4.795, abs=1e-3) and hi == pytest.approx(18.390, abs=1e-3)


def test_trailing_sigma_never_includes_the_day_it_judges():
    idx = pd.bdate_range("2000-01-03", periods=400)
    lr = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=idx)
    lr.iloc[300] = 0.06
    f = st.extreme_days(lr, 4.0, "trailing")
    assert f.loc[idx[300]]
    assert f.index[0] == idx[252]                      # first judged day has a full window
    with pytest.raises(ValueError):
        st.extreme_days(lr, 3.0, "sideways")


def test_fixed_sigma_count_is_gaussian_on_gaussian_data():
    idx = pd.bdate_range("1990-01-02", periods=60000)
    lr = pd.Series(np.random.default_rng(2).normal(0, 0.01, 60000), index=idx)
    rate = st.extreme_days(lr, 3.0, "fixed").mean()
    assert rate == pytest.approx(0.0027, abs=0.0008)


def test_count_table_rates_and_intervals():
    idx = pd.bdate_range("2000-01-03", "2009-12-31")
    f = pd.Series(False, index=idx)
    f.iloc[::100] = True
    t = st.count_table(f, (("2000s", 2000, 2004), ("late", 2005, 2009)))
    assert t["count"].sum() == f.sum()
    assert (t["ci_lo"] < t["rate"]).all() and (t["rate"] < t["ci_hi"]).all()
    assert t["rate"].iloc[0] == pytest.approx(10.0, rel=0.1)


def test_count_trend_sees_a_planted_rise_and_not_a_null(daily_null):
    f0, _ = daily_null
    f1, _ = data.synthetic_returns(n_years=30, freq="D", signal_strength=1.0,
                                   planted_multiple=4.0, logvol_sd=0.15, seed=1020)
    f0b, _ = data.synthetic_returns(n_years=30, freq="D", signal_strength=0.0,
                                    logvol_sd=0.15, seed=1020)
    up = st.count_trend(st.extreme_days(f1["logret"], 3.0, "fixed"), n_boot=299)
    flat = st.count_trend(st.extreme_days(f0b["logret"], 3.0, "fixed"), n_boot=299)
    assert up["slope"] > 0 and up["z_hac"] > 2 and up["p_perm"] < 0.05
    assert flat["p_perm"] > 0.05


def test_count_trend_on_an_empty_flag_series_is_inert():
    idx = pd.bdate_range("2000-01-03", "2009-12-31")
    out = st.count_trend(pd.Series(False, index=idx))
    assert out["slope"] == 0.0 and out["p_perm"] == 1.0


# --------------------------------------------------------------------------- #
# Points versus percent — the perception mechanism
# --------------------------------------------------------------------------- #
def test_a_constant_percent_index_makes_ever_bigger_point_moves():
    """The heart of the perception story: GBM with constant σ, a growing level."""
    f, _ = data.synthetic_returns(n_years=30, freq="D", signal_strength=0.0, logvol_sd=1e-9,
                                  drift=0.08, seed=7)
    pp = st.points_vs_percent(f["price"], big_points=10.0)
    assert st.passes(pp["trend_points"])
    assert not st.passes(pp["trend_pct"]) and not st.passes(pp["trend_pct"], -1)
    assert pp["top_points_share_last_decade"] > pp["top_pct_share_last_decade"]
    assert pp["days_over_big_second_half"] > pp["days_over_big_first_half"]
    # the identity: points trend ≈ level trend + percent trend
    s = pp["trend_points"]["slope"]
    assert s == pytest.approx(pp["trend_level"]["slope"] + pp["trend_pct"]["slope"], abs=0.01)


def test_records_are_strictly_new_lows():
    idx = pd.bdate_range("2000-01-03", periods=3000)
    p = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(8).normal(0.0003, 0.01, 3000))),
                  index=idx)
    pp = st.points_vs_percent(p, burn_years=2)
    rec = pp["records_points"]
    assert (rec < 0).all() and (np.diff(rec.to_numpy()) < 0).all()


# --------------------------------------------------------------------------- #
# The forecast race
# --------------------------------------------------------------------------- #
def test_qlike_is_zero_at_the_truth_and_positive_elsewhere():
    assert st.qlike(np.array([0.04]), np.array([0.04]))[0] == pytest.approx(0.0)
    assert (st.qlike(np.array([0.04, 0.04]), np.array([0.02, 0.08])) > 0).all()


def test_diebold_mariano_sign_convention():
    rng = np.random.default_rng(9)
    a = rng.uniform(0, 1, 300)
    dm = st.diebold_mariano(a, a + 0.5)
    assert dm["mean_diff"] == pytest.approx(-0.5) and dm["t"] < -10 and dm["p"] < 1e-6
    assert np.isnan(st.diebold_mariano(a[:5], a[:5])["t"])


def test_forecast_race_has_no_look_ahead(null_world):
    f, _ = null_world
    rv = st.annual_rv_from_monthly(f["ret"]) ** 2
    a = st.forecast_race(rv, recent=1, trend_window=20, burn=20)
    rv2 = rv.copy()
    rv2.iloc[50:] = rv2.iloc[50:] * 10.0          # change the future only
    b = st.forecast_race(rv2, recent=1, trend_window=20, burn=20)
    cols = ["LONG-RUN", "RECENT", "TREND", "WINDOW", "BLEND"]
    upto = a["forecasts"].index <= rv.index[50]      # forecast for period 50 uses <= 49
    assert np.allclose(a["forecasts"].loc[upto, cols], b["forecasts"].loc[upto, cols])
    assert a["n"] == len(rv) - 20


def test_trend_forecast_wins_only_where_a_trend_exists():
    """On a strongly trending world, extrapolation should beat the long-run mean."""
    f, _ = data.synthetic_returns(n_years=90, freq="M", signal_strength=1.0,
                                  planted_multiple=8.0, logvol_sd=0.15, seed=11)
    rv = st.annual_rv_from_monthly(f["ret"]) ** 2
    r = st.forecast_race(rv, recent=1, trend_window=20, burn=20)
    assert r["mean_qlike"]["TREND"] < r["mean_qlike"]["LONG-RUN"]


# --------------------------------------------------------------------------- #
# The verdict rule — both directions
# --------------------------------------------------------------------------- #
def _tr(slope, t_nw, p_kvb, n=91):
    return {"n": n, "slope": slope, "pct_per_decade": float(np.expm1(10 * slope)),
            "t_nw": t_nw, "t_kvb": t_nw * 1.3, "p_kvb": p_kvb, "p_boot": p_kvb}


def _race(trend, lr, recent, p=0.5, t=-1.0):
    dm = {k: {"mean_diff": (trend - lr) if "TREND" in k else -0.1, "t": t, "p": p}
          for k in ("TREND_vs_LONG-RUN", "TREND_vs_RECENT", "TREND_vs_WINDOW",
                    "RECENT_vs_LONG-RUN", "BLEND_vs_LONG-RUN")}
    return {"n": 70, "mean_qlike": {"LONG-RUN": lr, "RECENT": recent, "TREND": trend,
                                    "WINDOW": trend, "BLEND": recent}, "dm": dm}


def _h(**over):
    h = {"century": _tr(-0.0058, -2.1, 0.26), "daily": _tr(0.0063, 0.7, 0.61, 32),
         "ohlc": _tr(-0.03, -3.0, 0.05, 20),
         "extremes_fixed3": {"slope": 0.03, "z_hac": 1.5, "p_perm": 0.58, "total": 129,
                             "pct_per_decade": 0.4},
         "race_annual": _race(0.33, 0.47, 0.37), "race_monthly": _race(0.84, 0.68, 0.53),
         "thirties_vs_remembered": {"ratio": 2.24, "diff": 0.195, "p": 0.0003,
                                    "b": (2008, 2017)},
         "points": {"trend_points": {"pct_per_decade": 0.95},
                    "trend_pct": {"pct_per_decade": 0.04}},
         "century_span": "1927-2017", "daily_span": "1990-2021", "ohlc_span": "1999-2018",
         "share_starts_sig_up": 0.0, "starts_first": 1927, "starts_last": 1998,
         "century_last": 2017, "trend_window_annual": 20}
    h.update(over)
    return copy.deepcopy(h)


def test_verdict_signal_none_when_no_tape_trends_up():
    v = st.verdict(_h())
    assert v["signal"] == "None" and v["myth"] == "Busted"


def test_verdict_signal_real_when_century_and_daily_both_pass():
    v = st.verdict(_h(century=_tr(0.01, 3.0, 0.01), daily=_tr(0.02, 2.5, 0.02, 32)))
    assert v["signal"] == "Real" and v["myth"] == "Confirmed"


def test_verdict_signal_real_with_century_plus_extremes():
    v = st.verdict(_h(century=_tr(0.01, 3.0, 0.01),
                      extremes_fixed3={"slope": 0.05, "z_hac": 3.0, "p_perm": 0.01,
                                       "total": 129, "pct_per_decade": 0.6}))
    assert v["signal"] == "Real"


def test_verdict_signal_mixed_when_tapes_split():
    assert st.verdict(_h(daily=_tr(0.03, 2.8, 0.01, 32)))["signal"] == "Mixed"
    assert st.verdict(_h(century=_tr(0.01, 3.0, 0.01)))["signal"] == "Mixed"


def test_verdict_signal_weak_on_a_positive_but_unproven_slope():
    assert st.verdict(_h(daily=_tr(0.01, 1.4, 0.4, 32)))["signal"] == "Weak"
    assert st.verdict(_h(extremes_fixed3={"slope": 0.05, "z_hac": 3.0, "p_perm": 0.01,
                                          "total": 129, "pct_per_decade": 0.6}))["signal"] == "Weak"
    # a significant slope in the WRONG direction is not support for the claim
    assert st.verdict(_h(century=_tr(-0.02, -4.0, 0.001)))["signal"] == "None"


def test_verdict_tradability_all_three_stamps():
    assert st.verdict(_h())["trad"] == "Fragile"        # trend point-beats LR on one tape
    mir = _h(race_annual=_race(0.50, 0.47, 0.37))
    assert st.verdict(mir)["trad"] == "Mirage"
    inv = _h(race_annual=_race(0.20, 0.47, 0.37, p=0.01, t=-3.0),
             race_monthly=_race(0.30, 0.68, 0.53, p=0.01, t=-3.0))
    assert st.verdict(inv)["trad"] == "Investable"


def test_verdict_myth_not_supported_when_the_thirties_contrast_fails():
    v = st.verdict(_h(thirties_vs_remembered={"ratio": 1.1, "diff": 0.01, "p": 0.4,
                                              "b": (2008, 2017)}))
    assert v["myth"] == "Not supported"


def test_verdict_prose_is_complete():
    v = st.verdict(_h())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "myth", "one_sentence", "flags"}
    assert "KVB" in v["signal_why"] and "QLIKE" in v["trad_why"]
    assert "negative" in v["signal_why"]
    assert "points" in v["one_sentence"]


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_century_has_no_upward_trend():
    m = data.load_monthly()
    rv = st.annual_rv_from_monthly(m, data.FIRST_FULL_YEAR_MONTHLY, data.LAST_FULL_YEAR_MONTHLY)
    assert len(rv) == 91
    tr = st.trend_test(np.log(rv), n_boot=499)
    assert not st.passes(tr)


@needs_real
def test_real_thirties_dwarf_the_remembered_decade():
    dc = st.decade_contrast(data.load_monthly(), (1930, 1939), (2008, 2017), n_boot=1000)
    assert dc["ratio"] > 1.5 and dc["ci_lo"] > 0


@needs_real
def test_real_points_trend_is_the_level_not_the_risk():
    pp = st.points_vs_percent(data.load_daily())
    assert pp["trend_points"]["t_nw"] > 3
    assert abs(pp["trend_pct"]["t_nw"]) < 2
    assert pp["top_points_share_last_decade"] > 0.5 > pp["top_pct_share_last_decade"]


@needs_real
def test_real_range_and_close_estimators_agree_on_the_level():
    o = data.load_ohlc()
    ratio = (st.annual_rv_parkinson(o) / st.annual_rv_from_daily(o["close"])).mean()
    assert 0.6 < ratio < 1.1
