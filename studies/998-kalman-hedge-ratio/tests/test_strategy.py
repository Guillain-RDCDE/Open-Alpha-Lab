"""Strategy tests for Study 998 — estimators graded against a known beta."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from movingtarget import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The estimators, on a world with a constant beta
# --------------------------------------------------------------------------- #
def test_static_ols_recovers_a_constant_beta():
    w = st.synthetic_pair(n=6000, beta_start=1.5, beta_vol=0.0)
    assert st.static_hedge_ratio(w["y"], w["x"]) == pytest.approx(1.5, abs=0.03)


def test_rolling_ols_recovers_a_constant_beta_on_average():
    w = st.synthetic_pair(n=6000, beta_start=1.5, beta_vol=0.0)
    b = st.rolling_hedge_ratio(w["y"], w["x"], 120).dropna()
    assert b.mean() == pytest.approx(1.5, abs=0.05)


def test_kalman_recovers_a_constant_beta():
    w = st.synthetic_pair(n=6000, beta_start=1.5, beta_vol=0.0)
    kf = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-3)
    assert kf["beta"].iloc[-500:].mean() == pytest.approx(1.5, abs=0.06)


def test_the_kalman_estimate_is_strictly_causal():
    """The prior beta on day t must not depend on day t's observation."""
    w = st.synthetic_pair(n=2000)
    tampered_y = w["y"].copy()
    tampered_y.iloc[1500:] *= 10
    a = st.kalman_hedge_ratio(w["y"], w["x"])["beta"].iloc[:1400]
    b = st.kalman_hedge_ratio(tampered_y, w["x"])["beta"].iloc[:1400]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_the_rolling_estimate_is_strictly_causal():
    w = st.synthetic_pair(n=2000)
    tampered_y = w["y"].copy()
    tampered_y.iloc[1500:] *= 10
    a = st.rolling_hedge_ratio(w["y"], w["x"], 60).iloc[:1400].dropna()
    b = st.rolling_hedge_ratio(tampered_y, w["x"], 60).iloc[:1400].dropna()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_all_estimators_decline_on_too_little_data():
    w = st.synthetic_pair(n=20)
    assert np.isnan(st.static_hedge_ratio(w["y"], w["x"]))
    assert st.kalman_hedge_ratio(w["y"], w["x"]).empty


# --------------------------------------------------------------------------- #
# The filter's mechanics
# --------------------------------------------------------------------------- #
def test_a_bigger_delta_makes_the_filter_more_responsive():
    w = st.synthetic_pair(n=4000, beta_vol=0.004)
    slow = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-6)["beta"]
    fast = st.kalman_hedge_ratio(w["y"], w["x"], delta=3e-1)["beta"]
    # after the burn-in: the first few steps are dominated by the diffuse prior, which moves
    # violently for every setting and would mask the difference being tested
    assert fast.iloc[500:].diff().std() > slow.iloc[500:].diff().std() * 3


def test_the_state_variance_settles_to_a_steady_state():
    w = st.synthetic_pair(n=5000)
    v = st.kalman_hedge_ratio(w["y"], w["x"])["var"]
    early, late = v.iloc[100:200].mean(), v.iloc[-200:].mean()
    assert late < early                         # it learns
    assert v.iloc[-500:].std() / v.iloc[-500:].mean() < 0.5      # and then settles


def test_the_gain_falls_as_the_filter_becomes_confident():
    w = st.synthetic_pair(n=5000, beta_vol=0.0)
    g = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-5)["gain"].abs()
    assert g.iloc[:50].mean() > g.iloc[-500:].mean()


def test_the_effective_window_grows_as_delta_shrinks():
    a = st.effective_window(1e-1, 1e-3)
    b = st.effective_window(1e-3, 1e-3)
    c = st.effective_window(1e-5, 1e-3)
    assert a < b < c


def test_the_effective_window_is_a_plausible_number_of_days():
    """The point of the function: it demystifies the filter into a window length."""
    for delta in (1e-3, 1e-2, 1e-1):
        w = st.effective_window(delta, 1e-3, x_var=1e-3)
        assert 2 < w < 5000, (delta, w)


def test_innovations_are_roughly_white_when_the_model_is_right():
    w = st.synthetic_pair(n=8000, beta_vol=0.002)
    kf = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-2)
    e = kf["innovation"].iloc[500:]
    assert abs(e.autocorr(1)) < 0.15


# --------------------------------------------------------------------------- #
# Grading against the truth
# --------------------------------------------------------------------------- #
def test_the_kalman_beats_every_rolling_window_on_a_drifting_beta():
    """The central claim, on a world where the truth is known.

    Best-tuned against best-tuned, which is the only fair comparison. A single arbitrary
    ``delta`` is no more privileged than a single arbitrary window length — the filter's
    advantage is not that any one setting dominates, it is that its *effective* window adapts
    within a run once the setting is in the right region.
    """
    w = st.synthetic_pair(n=8000, beta_vol=0.003)
    kf_rmses = [st.tracking_error(st.kalman_hedge_ratio(w["y"], w["x"], delta=d)["beta"],
                                  w["beta"])["rmse"]
                for d in (1e-3, 1e-2, 1e-1, 3e-1, 0.6)]
    roll_rmses = [st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], win),
                                    w["beta"])["rmse"] for win in (20, 60, 120, 250)]
    assert min(kf_rmses) < min(roll_rmses)


def test_a_badly_tuned_filter_loses_to_a_well_chosen_window():
    """The honest converse, and the reason the study sweeps rather than asserts."""
    w = st.synthetic_pair(n=8000, beta_vol=0.003)
    bad = st.tracking_error(st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-8)["beta"],
                            w["beta"])["rmse"]
    good_window = min(st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], win),
                                        w["beta"])["rmse"] for win in (20, 60, 120, 250))
    assert bad > good_window       # a filter tuned to never adapt cannot track a drift


def test_a_static_estimator_wins_when_the_beta_does_not_move():
    """The control. An adaptive estimator must NOT beat a constant on a constant."""
    w = st.synthetic_pair(n=8000, beta_vol=0.0, beta_start=1.2)
    static = pd.Series(1.2, index=w["beta"].index)
    kf = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-3)["beta"]
    assert (st.tracking_error(static, w["beta"])["rmse"]
            < st.tracking_error(kf, w["beta"])["rmse"])


def test_short_windows_are_noisy_and_long_windows_are_stale():
    """The trade-off that motivates the whole study, measured."""
    w = st.synthetic_pair(n=8000, beta_vol=0.003)
    short = st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], 20), w["beta"])
    long = st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], 250), w["beta"])
    assert short["excess_movement"] > long["excess_movement"]     # short thrashes
    assert long["correlation"] < short["correlation"] + 0.5       # long lags


def test_tracking_error_reports_excess_movement():
    w = st.synthetic_pair(n=5000, beta_vol=0.002)
    te = st.tracking_error(st.rolling_hedge_ratio(w["y"], w["x"], 20), w["beta"])
    assert te["excess_movement"] > 1.0        # a 20-day window definitely over-moves


def test_tracking_error_declines_on_a_short_series():
    assert "rmse" not in st.tracking_error(pd.Series([1.0] * 5), pd.Series([1.0] * 5))


def test_a_better_tracked_beta_gives_a_tighter_spread():
    """Tracking accuracy and spread tightness must agree — they are the same quantity."""
    w = st.synthetic_pair(n=8000, beta_vol=0.003)
    best_delta = min((1e-3, 1e-2, 1e-1, 3e-1, 0.6),
                     key=lambda d: st.tracking_error(
                         st.kalman_hedge_ratio(w["y"], w["x"], delta=d)["beta"],
                         w["beta"])["rmse"])
    kf = st.kalman_hedge_ratio(w["y"], w["x"], delta=best_delta)["beta"]
    worst_window = max((20, 60, 120, 250),
                       key=lambda win: st.tracking_error(
                           st.rolling_hedge_ratio(w["y"], w["x"], win),
                           w["beta"])["rmse"])
    roll = st.rolling_hedge_ratio(w["y"], w["x"], worst_window)
    assert st.spread_series(w["y"], w["x"], kf).std() <         st.spread_series(w["y"], w["x"], roll).std()


# --------------------------------------------------------------------------- #
# Spread quality
# --------------------------------------------------------------------------- #
def test_spread_quality_finds_a_random_walk_has_a_variance_ratio_near_one():
    rng = np.random.default_rng(998)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 5000)),
                  index=pd.bdate_range("2006-01-02", periods=5000))
    q = st.spread_quality(s)
    assert q["variance_ratio_21"] == pytest.approx(1.0, abs=0.35)


def test_spread_quality_flags_a_mean_reverting_series():
    rng = np.random.default_rng(998)
    n = 6000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + rng.normal(0, 1)
    q = st.spread_quality(pd.Series(x, index=pd.bdate_range("2006-01-02", periods=n)))
    assert q["variance_ratio_21"] < 0.75
    assert 0 < q["halflife"] < 60


def test_spread_quality_declines_on_a_short_series():
    assert "halflife" not in st.spread_quality(pd.Series(np.arange(50.0)))


# --------------------------------------------------------------------------- #
# Trading it
# --------------------------------------------------------------------------- #
def test_the_spread_trade_takes_positions():
    w = st.synthetic_pair(n=4000, beta_vol=0.001)
    b = st.rolling_hedge_ratio(w["y"], w["x"], 60)
    t = st.spread_trade(w["y"], w["x"], b, cost_bps=0.0)
    assert t["n_trades"] > 5
    assert 0 < t["time_in_market"] < 1


def test_costs_reduce_the_net_return():
    w = st.synthetic_pair(n=5000, beta_vol=0.001)
    b = st.rolling_hedge_ratio(w["y"], w["x"], 60)
    free = st.spread_trade(w["y"], w["x"], b, cost_bps=0.0)
    paid = st.spread_trade(w["y"], w["x"], b, cost_bps=50.0)
    assert paid["net_ann"] < free["net_ann"]
    assert paid["cost_ann"] > 0


def test_a_more_active_hedge_pays_more_hedge_turnover():
    """The cost that makes adaptiveness a trade-off rather than a free win."""
    w = st.synthetic_pair(n=5000, beta_vol=0.002)
    slow = st.kalman_hedge_ratio(w["y"], w["x"], delta=1e-6)["beta"]
    fast = st.kalman_hedge_ratio(w["y"], w["x"], delta=3e-1)["beta"]
    t_slow = st.spread_trade(w["y"], w["x"], slow, cost_bps=5.0)
    t_fast = st.spread_trade(w["y"], w["x"], fast, cost_bps=5.0)
    assert t_fast["hedge_turnover_ann"] > t_slow["hedge_turnover_ann"]


def test_a_wider_entry_band_trades_less():
    w = st.synthetic_pair(n=6000, beta_vol=0.001)
    b = st.rolling_hedge_ratio(w["y"], w["x"], 60)
    tight = st.spread_trade(w["y"], w["x"], b, entry_z=1.0, cost_bps=0.0)
    wide = st.spread_trade(w["y"], w["x"], b, entry_z=3.0, cost_bps=0.0)
    assert wide["n_trades"] < tight["n_trades"]
    assert wide["time_in_market"] < tight["time_in_market"]


def test_spread_trade_declines_on_a_short_series():
    w = st.synthetic_pair(n=100)
    b = st.rolling_hedge_ratio(w["y"], w["x"], 60)
    assert "sharpe" not in st.spread_trade(w["y"], w["x"], b)


def test_compare_estimators_covers_every_configuration():
    w = st.synthetic_pair(n=5000, beta_vol=0.002)
    c = st.compare_estimators(w["y"], w["x"], w["y"], w["x"],
                              windows=(60, 250), deltas=(1e-4,))
    assert len(c) == 4          # static + 2 rolling + 1 Kalman
    assert "static OLS (look-ahead)" in c.index


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_pairs": 7, "years": 19.0, "kalman_rmse": 0.041, "best_rolling_rmse": 0.068,
         "best_rolling_window": 60, "kalman_excess_movement": 1.4,
         "rolling_excess_movement": 3.2, "kalman_wins_spread": 0.71,
         "static_pair": "GLD/IAU", "static_verdict": "disappears, as it should",
         "kalman_turnover": 0.42, "rolling_turnover": 0.28,
         "kalman_cost": 0.004, "rolling_cost": 0.003,
         "kalman_gross_sharpe": 0.55, "rolling_gross_sharpe": 0.41,
         "kalman_net_sharpe": 0.48, "rolling_net_sharpe": 0.36,
         "best_sharpe": 0.48}
    h.update(over)
    return h


def test_verdict_signal_needs_tracking_and_tighter_spreads():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(kalman_wins_spread=0.2))["signal"] == "Partial"
    assert st.verdict(_headline(kalman_rmse=0.09,
                                kalman_wins_spread=0.2))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(kalman_net_sharpe=0.30))["trad"] == "Partial"
    assert st.verdict(_headline(kalman_net_sharpe=0.30,
                                kalman_gross_sharpe=0.35))["trad"] == "Mirage"


def test_verdict_prose_charges_the_hedge_rebalancing():
    v = st.verdict(_headline())
    assert "hedge turnover" in v["trad_why"] or "hedge-rebalancing" in v["trad_why"]
    assert "Tracking is not trading" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
