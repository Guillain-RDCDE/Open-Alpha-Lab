"""Strategy tests for Study 964 — offline, deterministic, and adversarial where it counts.

The mechanics (running peak, drawdown, one-day lag, cost accounting) are pinned against
hand-checkable cases; the inference is pinned against a random-walk null, where a record
high must carry no information at all.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ath_buy import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The state variables
# --------------------------------------------------------------------------- #
def test_running_peak_and_drawdown_on_a_hand_case():
    px = pd.Series([100.0, 110.0, 99.0, 105.0, 121.0],
                   index=pd.bdate_range("2020-01-01", periods=5))
    assert list(st.running_peak(px)) == [100, 110, 110, 110, 121]
    dd = st.drawdown(px)
    assert dd.iloc[0] == 0.0 and dd.iloc[1] == 0.0
    assert dd.iloc[2] == pytest.approx(99 / 110 - 1)
    assert (dd <= 1e-15).all()


def test_at_high_is_exactly_the_running_max():
    px = pd.Series([1.0, 2.0, 1.5, 2.0, 3.0], index=pd.bdate_range("2020-01-01", periods=5))
    assert list(st.at_high(px)) == [True, True, False, True, True]


def test_tolerance_widens_the_definition():
    px = pd.Series([100.0, 100.0, 99.6], index=pd.bdate_range("2020-01-01", periods=3))
    assert not st.at_high(px).iloc[2]
    assert st.at_high(px, tol=0.005).iloc[2]


def test_forward_return_is_forward_and_aligned():
    px = pd.Series(np.arange(1, 11, dtype=float), index=pd.bdate_range("2020-01-01", periods=10))
    f = st.forward_return(px, 3)
    assert f.iloc[0] == pytest.approx(4 / 1 - 1)
    assert np.isnan(f.iloc[-3:]).all()


# --------------------------------------------------------------------------- #
# Conditional statistics
# --------------------------------------------------------------------------- #
def _trending(prices):
    """The column that actually rose — a tape that never made a new high has no sample."""
    return prices.loc[:, prices.iloc[-1].idxmax()]


def test_forward_table_shape_and_horizons(planted):
    prices, cash, _ = planted
    tbl = st.forward_table(_trending(prices))
    assert list(tbl.index) == list(st.HORIZONS)
    assert (tbl["n_state"] > 0).all()
    assert np.isfinite(tbl["t_diff"]).all()


def test_hac_mean_caps_the_lag_and_never_returns_nan(planted):
    """252 lags on a few hundred observations is not estimable — the cap must bite."""
    prices, _, _ = planted
    px = _trending(prices)
    f = st.forward_return(px, 252).dropna()
    m = st.at_high(px).reindex(f.index).fillna(False)
    r = st.hac_mean(f[m], 252)
    assert r["lags_used"] == min(252, len(f[m]) // 4)
    assert np.isfinite(r["tstat"])
    # And the cap is the only thing that changed: with a short series the raw estimator
    # is entitled to give up, and this wrapper is entitled to fall back.
    tiny = st.hac_mean(f[m].iloc[:8], 252)
    assert tiny["lags_used"] <= 2 and np.isfinite(tiny["tstat"])


def test_conditional_stats_t_matches_its_own_components(planted):
    prices, _, _ = planted
    px = _trending(prices)
    fwd, mask = st.forward_return(px, 63), st.at_high(px)
    out = st.conditional_stats(fwd, mask, 63)
    f = fwd.dropna()
    m = mask.reindex(f.index).fillna(False)
    a, b = st.hac_mean(f[m], 63), st.hac_mean(f[~m], 63)
    se = np.sqrt(a["se_bps"] ** 2 + b["se_bps"] ** 2) / 1e4
    assert out["t_diff"] == pytest.approx(out["diff"] / se)


def test_nonoverlap_stats_uses_far_fewer_observations(planted):
    prices, _, _ = planted
    px = _trending(prices)
    over = st.conditional_stats(st.forward_return(px, 252), st.at_high(px), 252)
    non = st.nonoverlap_stats(px, 252)
    assert non["n_state"] + non["n_other"] <= (over["n_state"] + over["n_other"]) / 200 + 5


def test_random_walk_null_shows_no_conditional_edge():
    """On a pure random walk with drift, a record high must not predict anything.

    Twenty seeds; the mean |t| of the 12-month gap must stay small and hits must be rare.
    """
    ts = []
    for s in range(20):
        rng = np.random.default_rng(964 + s)
        r = rng.normal(0.0003, 0.011, 4000)
        px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("1995-01-02", periods=4000))
        ts.append(st.conditional_stats(st.forward_return(px, 252), st.at_high(px), 252)["t_diff"])
    ts = np.abs(np.array(ts))
    assert np.nanmean(ts) < 2.0
    assert (ts >= 2.0).mean() <= 0.35


def test_drawdown_buckets_partition_the_sample(planted):
    prices, _, _ = planted
    px = _trending(prices)
    tbl = st.drawdown_bucket_table(px, horizon=63)
    assert tbl["n"].sum() <= len(st.forward_return(px, 63).dropna()) + 1
    assert (tbl["n"] > 0).any()
    assert "at the high" in tbl.index


# --------------------------------------------------------------------------- #
# The strategy: lag, costs, and the arithmetic of being out of the market
# --------------------------------------------------------------------------- #
def test_dip_strategy_has_exactly_one_day_of_lag():
    """A crash on day t can only be acted on from day t+1."""
    px = pd.Series([100, 100, 100, 80, 80, 80, 80.0],
                   index=pd.bdate_range("2020-01-01", periods=7))
    cash = pd.Series(1.0, index=px.index)
    f = st.dip_strategy(px, cash, dip_pct=0.05, cost_bps=0.0)
    # Day 3 (index 3) is the crash: the drawdown is only known at its close, so the
    # position is still cash that day and only becomes invested on day 4.
    assert f["invested"].iloc[3] == 0
    assert f["invested"].iloc[4] == 1


def test_costs_are_charged_on_switches_only():
    px = pd.Series([100, 100, 90, 90, 100, 100.0],
                   index=pd.bdate_range("2020-01-01", periods=6))
    cash = pd.Series(1.0, index=px.index)
    free = st.dip_strategy(px, cash, dip_pct=0.05, cost_bps=0.0)["strategy"]
    paid = st.dip_strategy(px, cash, dip_pct=0.05, cost_bps=10.0)["strategy"]
    charged = (free - paid)
    n_switches = int(st.dip_strategy(px, cash, 0.05, 0.0)["invested"].diff().abs().sum())
    assert charged.sum() > 0
    assert np.allclose(charged[charged > 1e-12].to_numpy(), 10.0 / 1e4)
    assert (charged > 1e-12).sum() == n_switches


def test_strategy_equals_buy_and_hold_when_always_invested():
    px = pd.Series(np.linspace(100, 50, 300), index=pd.bdate_range("2020-01-01", periods=300))
    cash = pd.Series(1.0, index=px.index)
    f = st.dip_strategy(px, cash, dip_pct=0.0, cost_bps=0.0)
    # A monotonically falling tape is always at least 0% below its peak -> always invested
    # after the first day, so the two legs must agree from then on.
    assert np.allclose(f["strategy"].iloc[2:], f["buy_hold"].iloc[2:])


def test_performance_matches_a_hand_computed_curve():
    r = pd.Series([0.01] * 252, index=pd.bdate_range("2020-01-01", periods=252))
    p = st.performance(r)
    assert p["cagr"] == pytest.approx(1.01 ** 252 - 1, rel=1e-6)
    assert p["max_dd"] == pytest.approx(0.0)
    assert p["vol"] == pytest.approx(0.0, abs=1e-12)


def test_race_and_sweep_are_consistent(planted):
    prices, cash, _ = planted
    px, c = _trending(prices), cash
    r = st.race(px, c, dip_pct=0.05)
    sw = st.dip_sweep(px, c)
    assert sw.loc[0.05, "cagr_gap"] == pytest.approx(r["cagr_gap"])
    assert (sw["time_invested"].diff().dropna() <= 1e-9).all()  # deeper dip -> less time in


def test_deeper_dip_means_less_time_invested_and_fewer_switches(planted):
    prices, cash, _ = planted
    sw = st.dip_sweep(_trending(prices), cash, dips=(0.02, 0.10, 0.25))
    assert sw.loc[0.02, "time_invested"] > sw.loc[0.25, "time_invested"]


def test_share_of_days_at_high_is_a_fraction(planted):
    prices, _, _ = planted
    s = st.share_of_days_at_high(_trending(prices))
    assert 0.0 <= s <= 1.0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"pooled_t_12m": 0.4, "pooled_diff_12m": 0.01, "nonoverlap_diff_12m": 0.01,
         "n_positive_12m": 3, "pooled_state_12m": 0.11, "pooled_other_12m": 0.10,
         "share_at_high_spy": 0.07, "tickers": list("ABCDEF"), "n_dip_sharpe_wins": 1,
         "pooled_dip_t": 0.5, "head_dip": 0.05, "head_time_invested": 0.4,
         "head_cagr_gap": -0.04, "head_sharpe_gap": -0.3, "head_t_gap": -1.5}
    h.update(over)
    return h


def test_verdict_none_when_nothing_clears():
    v = st.verdict(_headline())
    assert v["signal"] == "None" and v["trad"] == "Mirage"


def test_verdict_weak_on_a_consistent_sign_without_significance():
    assert st.verdict(_headline(n_positive_12m=5))["signal"] == "Weak"


def test_verdict_real_needs_both_the_t_and_the_nonoverlap_sign():
    assert st.verdict(_headline(pooled_t_12m=3.0))["signal"] == "Real"
    assert st.verdict(_headline(pooled_t_12m=3.0, nonoverlap_diff_12m=-0.02))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline(n_dip_sharpe_wins=5, pooled_dip_t=2.5))["trad"] == "Investable"
    assert st.verdict(_headline(n_dip_sharpe_wins=5, pooled_dip_t=0.5))["trad"] == "Fragile"
    assert st.verdict(_headline(n_dip_sharpe_wins=2))["trad"] == "Mirage"
