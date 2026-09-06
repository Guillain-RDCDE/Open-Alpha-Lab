"""Strategy tests for Study 999 — detectors graded against known break dates."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from thebreak import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# CUSUM mechanics
# --------------------------------------------------------------------------- #
def test_cusum_never_goes_negative():
    w = st.synthetic_series(n=2000)
    c = st.cusum(w["returns"])
    assert (c["pos"] >= 0).all() and (c["neg"] >= 0).all()


def test_cusum_stays_quiet_on_pure_noise():
    rng = np.random.default_rng(999)
    x = pd.Series(rng.normal(0, 1, 5000),
                  index=pd.bdate_range("1993-02-01", periods=5000))
    c = st.cusum(x, drift=0.5, threshold=8.0)
    assert st.alarm_rate(c["alarm"])["alarms_per_year"] < 3


def test_cusum_fires_on_a_planted_mean_shift():
    rng = np.random.default_rng(999)
    n = 3000
    x = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(1.5, 1, n // 2)])
    s = pd.Series(x, index=pd.bdate_range("1993-02-01", periods=n))
    c = st.cusum(s, drift=0.5, threshold=5.0)
    fired = c.index[c["alarm"]]
    assert len(fired) > 0
    assert (fired >= s.index[n // 2]).any()


def test_a_bigger_shift_is_detected_faster():
    """The information bound, visible: more evidence per observation means less waiting."""
    delays = []
    for shift in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(999)
        n = 4000
        x = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(shift, 1, n // 2)])
        s = pd.Series(x, index=pd.bdate_range("1993-02-01", periods=n))
        c = st.cusum(s, drift=0.25, threshold=6.0)
        dd = st.detection_delay(c["alarm"], [s.index[n // 2]])
        delays.append(dd["median_delay"])
    valid = [d for d in delays if np.isfinite(d)]
    assert len(valid) >= 2
    assert valid[0] > valid[-1]


def test_a_higher_threshold_means_fewer_alarms():
    w = st.synthetic_series(n=5000)
    low = st.alarm_rate(st.variance_cusum(w["returns"], threshold=3.0)["alarm"])
    high = st.alarm_rate(st.variance_cusum(w["returns"], threshold=15.0)["alarm"])
    assert high["alarms_per_year"] < low["alarms_per_year"]


def test_a_higher_drift_parameter_also_quietens_it():
    w = st.synthetic_series(n=5000)
    lo = st.alarm_rate(st.variance_cusum(w["returns"], drift=0.1, threshold=5.0)["alarm"])
    hi = st.alarm_rate(st.variance_cusum(w["returns"], drift=1.5, threshold=5.0)["alarm"])
    assert hi["alarms_per_year"] <= lo["alarms_per_year"]


def test_the_statistic_resets_after_an_alarm():
    w = st.synthetic_series(n=3000, vol_shift=4.0)
    c = st.variance_cusum(w["returns"], threshold=4.0)
    fired = np.flatnonzero(c["alarm"].to_numpy())
    if len(fired):
        i = int(fired[0])
        assert c["pos"].iloc[i] == 0.0 and c["neg"].iloc[i] == 0.0


def test_cusum_uses_only_the_warmup_for_standardisation():
    """No look-ahead: tampering with the tail must not change the head."""
    w = st.synthetic_series(n=3000)
    r = w["returns"]
    bad = r.copy()
    bad.iloc[2000:] *= 20
    a = st.cusum(r, warmup=250)["pos"].iloc[:1800]
    b = st.cusum(bad, warmup=250)["pos"].iloc[:1800]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_cusum_declines_on_a_short_series():
    w = st.synthetic_series(n=100)
    assert st.cusum(w["returns"]).empty


# --------------------------------------------------------------------------- #
# The variance detector
# --------------------------------------------------------------------------- #
def test_variance_cusum_finds_a_volatility_regime_change():
    w = st.synthetic_series(n=4000, break_points=[2000], vol_shift=3.0)
    c = st.variance_cusum(w["returns"], threshold=5.0)
    dd = st.detection_delay(c["alarm"], w["breaks"])
    assert dd["n_detected"] == 1
    assert dd["median_delay"] < 250


def test_variance_cusum_ignores_a_pure_mean_shift_it_was_not_asked_about():
    """A volatility detector should be largely blind to a small mean change."""
    w_mean = st.synthetic_series(n=4000, break_points=[2000], vol_shift=1.0,
                                 mean_shift=0.5)
    w_vol = st.synthetic_series(n=4000, break_points=[2000], vol_shift=3.0)
    a = st.alarm_rate(st.variance_cusum(w_mean["returns"], threshold=6.0)["alarm"])
    b = st.alarm_rate(st.variance_cusum(w_vol["returns"], threshold=6.0)["alarm"])
    assert a["alarms_per_year"] < b["alarms_per_year"]


def test_a_bigger_volatility_shift_is_found_sooner():
    delays = []
    for vs in (1.5, 3.0, 6.0):
        w = st.synthetic_series(n=4000, break_points=[2000], vol_shift=vs)
        dd = st.detection_delay(st.variance_cusum(w["returns"], threshold=5.0)["alarm"],
                                w["breaks"])
        delays.append(dd["median_delay"])
    valid = [d for d in delays if np.isfinite(d)]
    assert len(valid) >= 2
    assert valid[0] > valid[-1]


# --------------------------------------------------------------------------- #
# Retrospective segmentation
# --------------------------------------------------------------------------- #
def test_binary_segmentation_finds_a_planted_break_accurately():
    w = st.synthetic_series(n=4000, break_points=[2000], vol_shift=3.0)
    breaks = st.binary_segmentation(w["returns"], max_breaks=3)
    assert len(breaks) >= 1
    truth = w["breaks"][0]
    nearest = min(breaks, key=lambda b: abs((b - truth).days))
    assert abs((nearest - truth).days) < 120


def test_binary_segmentation_finds_nothing_in_a_homogeneous_series():
    rng = np.random.default_rng(999)
    x = pd.Series(rng.normal(0, 0.01, 4000),
                  index=pd.bdate_range("1993-02-01", periods=4000))
    assert len(st.binary_segmentation(x, max_breaks=5)) <= 1


def test_binary_segmentation_respects_the_break_budget():
    w = st.synthetic_series(n=6000, break_points=[1000, 2000, 3000, 4000, 5000],
                            vol_shift=3.0)
    assert len(st.binary_segmentation(w["returns"], max_breaks=2)) <= 2


def test_a_bigger_penalty_finds_fewer_breaks():
    w = st.synthetic_series(n=5000, break_points=[1200, 2400, 3600], vol_shift=2.0)
    few = st.binary_segmentation(w["returns"], max_breaks=6, penalty=1e6)
    many = st.binary_segmentation(w["returns"], max_breaks=6, penalty=1.0)
    assert len(few) <= len(many)


def test_segmentation_declines_on_a_short_series():
    w = st.synthetic_series(n=80)
    assert st.binary_segmentation(w["returns"]) == []


def test_retrospective_beats_sequential_by_construction():
    """The study's spine: hindsight is earlier than real time, always."""
    w = st.synthetic_series(n=5000, break_points=[2500], vol_shift=3.0)
    seq = st.detection_delay(st.variance_cusum(w["returns"], threshold=5.0)["alarm"],
                             w["breaks"])
    retro = st.binary_segmentation(w["returns"], max_breaks=2)
    truth = w["breaks"][0]
    if retro and np.isfinite(seq["median_delay"]):
        retro_err = min(abs((b - truth).days) for b in retro)
        assert retro_err < seq["median_delay"] * 1.6 + 30


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_detection_delay_counts_only_alarms_after_the_break():
    idx = pd.bdate_range("2020-01-01", periods=300)
    alarms = pd.Series(False, index=idx)
    alarms.iloc[50] = True        # before the break: a false alarm
    alarms.iloc[150] = True       # after: the detection
    dd = st.detection_delay(alarms, [idx[100]])
    assert dd["n_detected"] == 1
    assert dd["false_alarms"] == 1
    assert dd["median_delay"] == pytest.approx(
        float(np.busday_count(idx[100].date(), idx[150].date())))


def test_a_break_that_is_never_detected_scores_nan():
    idx = pd.bdate_range("2020-01-01", periods=300)
    alarms = pd.Series(False, index=idx)
    dd = st.detection_delay(alarms, [idx[100]])
    assert dd["n_detected"] == 0
    assert np.isnan(dd["median_delay"])


def test_a_very_late_detection_does_not_count():
    idx = pd.bdate_range("2020-01-01", periods=1200)
    alarms = pd.Series(False, index=idx)
    alarms.iloc[1100] = True
    dd = st.detection_delay(alarms, [idx[100]], max_delay=200)
    assert dd["n_detected"] == 0


def test_alarm_rate_scales_with_the_series_length():
    idx = pd.bdate_range("2000-01-01", periods=2520)
    a = pd.Series(False, index=idx)
    a.iloc[::252] = True                          # one a year
    assert st.alarm_rate(a)["alarms_per_year"] == pytest.approx(1.0, abs=0.05)


def test_alarm_rate_declines_on_a_short_series():
    assert "alarms_per_year" not in st.alarm_rate(pd.Series([False] * 20))


def test_the_roc_curve_trades_delay_against_false_alarms():
    w = st.synthetic_series(n=6000, break_points=[2000, 4000], vol_shift=3.0)
    roc = st.roc_curve(w["returns"], w["breaks"], thresholds=(3, 6, 12))
    assert roc["alarms_per_year"].is_monotonic_decreasing
    valid = roc["median_delay"].dropna()
    assert len(valid) >= 2


def test_the_theoretical_delay_shrinks_with_the_change_size():
    assert st.theoretical_delay(1.0, 5.0) > st.theoretical_delay(3.0, 5.0)
    assert st.theoretical_delay(2.0, 10.0) > st.theoretical_delay(2.0, 5.0)


def test_a_change_smaller_than_the_drift_is_never_detected():
    """The honest limit: a CUSUM tuned for 0.5 sigma cannot see a 0.3 sigma shift."""
    assert np.isinf(st.theoretical_delay(0.3, 5.0, drift=0.5))


# --------------------------------------------------------------------------- #
# Using it
# --------------------------------------------------------------------------- #
def test_the_switching_rule_spends_time_out_of_the_market():
    w = st.synthetic_series(n=5000, vol_shift=3.0)
    c = st.variance_cusum(w["returns"], threshold=5.0)
    out = st.regime_switch_strategy(w["returns"], c["alarm"])
    assert 0 < out["time_in_market"] < 1
    assert out["n_alarms"] > 0


def test_no_alarms_means_the_rule_is_just_buy_and_hold():
    w = st.synthetic_series(n=3000)
    never = pd.Series(False, index=w["returns"].index)
    out = st.regime_switch_strategy(w["returns"], never, cost_bps=0.0)
    assert out["strategy"]["cagr"] == pytest.approx(out["buy_hold"]["cagr"])
    assert out["time_in_market"] == pytest.approx(1.0)


def test_costs_reduce_the_switching_rule():
    w = st.synthetic_series(n=5000, vol_shift=3.0)
    c = st.variance_cusum(w["returns"], threshold=5.0)
    free = st.regime_switch_strategy(w["returns"], c["alarm"], cost_bps=0.0)
    paid = st.regime_switch_strategy(w["returns"], c["alarm"], cost_bps=100.0)
    assert paid["strategy"]["cagr"] < free["strategy"]["cagr"]


def test_hindsight_avoids_worse_days_per_alarm_than_the_live_rule():
    """The comparison has to be per-alarm, not per-strategy.

    A naive "hindsight beats live" assertion is wrong and it took a failing test to see why: the
    live detector fires far more often than there are true breaks, so in a bad enough world it
    can end up out of the market more and score better in total while being worse at the thing
    being measured. What perfect knowledge actually buys is *better-targeted* avoidance, so the
    honest comparison is the average return avoided per risk-off day.
    """
    # The planted drift has to be large enough to survive its own noise. At a 3% daily
    # volatility over 1,500 sessions the sample mean has a standard error near 0.08% a day, so
    # a -0.05% drift is invisible in-sample and the "bad" regime can come out positive by luck.
    # That is a real lesson about regime studies and a terrible foundation for an assertion.
    w = st.synthetic_series(n=8000, break_points=[2000, 4000, 6000], vol_shift=3.0,
                            mean_shift=-0.4)
    r = w["returns"]
    assert r[w["regime_high"]].mean() < r[~w["regime_high"]].mean()   # premise check
    c = st.variance_cusum(r, threshold=5.0)
    live = st.regime_switch_strategy(r, c["alarm"], cost_bps=0.0)
    hind = st.hindsight_strategy(r, w["bad_breaks"], cost_bps=0.0)

    def avoided(res):
        out = res["returns"].index[(res["returns"] - r.reindex(res["returns"].index)).abs()
                                   > 1e-12]
        return float(r.reindex(out).mean()) if len(out) else 0.0

    assert avoided(hind) <= avoided(live) + 1e-6
    assert 0 < hind["time_in_market"] <= 1.0


def test_only_half_the_breaks_lead_into_the_bad_regime():
    """The generator says which, because a hindsight benchmark that ignores it is worthless."""
    w = st.synthetic_series(n=6000, break_points=[1500, 3000, 4500], vol_shift=3.0)
    assert len(w["bad_breaks"]) == 2          # breaks 1 and 3 enter the turbulent regime
    assert set(w["bad_breaks"]) <= set(w["breaks"])
    for b in w["bad_breaks"]:
        assert w["regime_high"].loc[b]


def test_hindsight_wins_outright_when_the_bad_regime_is_short():
    """Where the fixed-duration rule can actually cover the regime, foresight wins cleanly."""
    w = st.synthetic_series(n=6000, break_points=[3000, 3020], vol_shift=5.0,
                            mean_shift=-0.15)
    r = w["returns"]
    c = st.variance_cusum(r, threshold=5.0)
    live = st.regime_switch_strategy(r, c["alarm"], cost_bps=0.0)
    hind = st.hindsight_strategy(r, w["bad_breaks"], cost_bps=0.0)
    assert hind["strategy"]["cagr"] >= live["strategy"]["cagr"] - 1e-9


def test_a_longer_risk_off_period_means_less_time_invested():
    w = st.synthetic_series(n=5000, vol_shift=3.0)
    c = st.variance_cusum(w["returns"], threshold=5.0)
    short = st.regime_switch_strategy(w["returns"], c["alarm"], risk_off_days=5)
    long = st.regime_switch_strategy(w["returns"], c["alarm"], risk_off_days=60)
    assert long["time_in_market"] < short["time_in_market"]


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"threshold": 5.0, "detection_rate": 0.83, "median_delay": 34.0,
         "alarms_per_year": 1.4, "theoretical_delay": 22.0, "delay_vs_theory": 1.55,
         "low_threshold": 3.0, "low_delay": 19.0, "low_alarm_rate": 3.8,
         "retro_advantage": 27.0, "risk_off_days": 21,
         "live_cagr": 0.061, "live_sharpe": 0.44, "live_dd": -0.38,
         "bh_cagr": 0.083, "bh_sharpe": 0.48, "bh_dd": -0.55,
         "hindsight_cagr": 0.121, "hindsight_sharpe": 0.79}
    h.update(over)
    return h


def test_verdict_signal_needs_detection_quiet_and_near_optimal():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(alarms_per_year=6.0))["signal"] == "Partial"
    assert st.verdict(_headline(delay_vs_theory=5.0))["signal"] == "Partial"
    assert st.verdict(_headline(detection_rate=0.2))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Partial"      # better DD, worse Sharpe
    assert st.verdict(_headline(live_sharpe=0.60))["trad"] == "Useful"
    assert st.verdict(_headline(live_dd=-0.60))["trad"] == "Mirage"


def test_verdict_prose_attributes_the_gap_to_the_delay():
    v = st.verdict(_headline())
    assert "price of the" in v["trad_why"]
    assert "theoretical" in v["signal_why"] or "Wald" in v["signal_why"]
    assert "late" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
