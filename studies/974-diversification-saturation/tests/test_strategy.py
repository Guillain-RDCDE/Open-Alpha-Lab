"""Strategy tests for Study 974 — the diversification curve against its closed form."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from diversify_n import data, strategy as st  # noqa: E402


def _factor_panel(n_assets=8, rho=0.3, vol=0.20, n=4000, seed=974):
    """Equicorrelated returns with a known correlation — the closed form applies exactly."""
    rng = np.random.default_rng(seed)
    sd = vol / np.sqrt(252)
    f = rng.normal(0, sd * np.sqrt(rho), (n, 1))
    idio = rng.normal(0, sd * np.sqrt(1 - rho), (n, n_assets))
    r = f + idio
    return pd.DataFrame(r, index=pd.bdate_range("2005-01-03", periods=n),
                        columns=[f"A{i}" for i in range(n_assets)])


# --------------------------------------------------------------------------- #
# The mechanics
# --------------------------------------------------------------------------- #
def test_portfolio_stats_matches_a_hand_case():
    r = pd.DataFrame({"A": [0.01] * 252, "B": [0.01] * 252},
                     index=pd.bdate_range("2020-01-01", periods=252))
    s = st.portfolio_stats(r, cost_bps=0.0)
    assert s["vol"] == pytest.approx(0.0, abs=1e-12)
    assert s["max_dd"] == pytest.approx(0.0, abs=1e-12)
    assert s["n_assets"] == 2


def test_costs_and_turnover_only_appear_at_rebalances():
    r = _factor_panel(n_assets=4, n=500)
    free = st.portfolio_stats(r, cost_bps=0.0)
    paid = st.portfolio_stats(r, cost_bps=100.0)
    assert free["turnover"] > 0
    assert paid["sharpe"] < free["sharpe"]
    never = st.portfolio_stats(r, rebalance=10 ** 6, cost_bps=100.0)
    assert never["turnover"] == 0.0


def test_two_identical_assets_diversify_nothing():
    r = _factor_panel(n_assets=1, n=2000)
    both = pd.concat([r["A0"].rename("A"), r["A0"].rename("B")], axis=1)
    one = st.portfolio_stats(both[["A"]], cost_bps=0.0)["vol"]
    two = st.portfolio_stats(both, cost_bps=0.0)["vol"]
    assert two == pytest.approx(one, rel=0.02)


def test_independent_assets_diversify_like_one_over_root_k():
    r = _factor_panel(n_assets=9, rho=0.0, n=6000)
    v1 = st.portfolio_stats(r[["A0"]], cost_bps=0.0)["vol"]
    v9 = st.portfolio_stats(r, cost_bps=0.0)["vol"]
    assert v9 == pytest.approx(v1 / 3, rel=0.15)


# --------------------------------------------------------------------------- #
# The curve
# --------------------------------------------------------------------------- #
def test_empirical_curve_matches_the_closed_form():
    r = _factor_panel(n_assets=8, rho=0.3, n=5000)
    emp = st.random_subset_curve(r, draws=60, cost_bps=0.0)
    th = st.theoretical_curve(r)
    for k in (1, 2, 4, 8):
        assert emp.loc[k, "vol_mean"] == pytest.approx(th.loc[k, "vol_theory"], rel=0.10)


def test_the_curve_is_monotone_and_converges_to_the_floor():
    r = _factor_panel(n_assets=10, rho=0.4, n=5000)
    emp = st.random_subset_curve(r, draws=50, cost_bps=0.0)
    assert emp["vol_mean"].is_monotonic_decreasing
    th = st.theoretical_curve(r)
    assert emp["vol_mean"].iloc[-1] > th.attrs["floor_vol"] * 0.9


def test_correlation_sets_the_floor():
    lo = st.theoretical_curve(_factor_panel(rho=0.1, n=4000))
    hi = st.theoretical_curve(_factor_panel(rho=0.7, n=4000))
    assert hi.attrs["floor_vol"] > lo.attrs["floor_vol"] * 2
    assert hi.attrs["avg_corr"] > lo.attrs["avg_corr"]


def test_marginal_benefit_and_stopping_point_agree():
    r = _factor_panel(n_assets=10, rho=0.3, n=5000)
    curve = st.random_subset_curve(r, draws=50, cost_bps=0.0)
    mb = st.marginal_benefit(curve, threshold=0.05)
    stop = st.stopping_point(curve, threshold=0.05)
    assert bool(mb.loc[stop, "worth_it"])
    assert 1 <= stop <= 10
    assert st.stopping_point(curve, threshold=0.01) >= stop


def test_dispersion_shrinks_as_the_portfolio_grows():
    r = _factor_panel(n_assets=10, rho=0.3, n=4000)
    curve = st.random_subset_curve(r, draws=80, cost_bps=0.0)
    spread = curve["vol_p90"] - curve["vol_p10"]
    assert spread.iloc[0] > spread.iloc[-2]


# --------------------------------------------------------------------------- #
# Effective bets and the greedy order
# --------------------------------------------------------------------------- #
def test_effective_bets_is_one_for_duplicates_and_n_for_independents():
    r = _factor_panel(n_assets=1, n=3000)
    dup = pd.concat([r["A0"].rename(f"C{i}") for i in range(5)], axis=1)
    assert st.effective_number_of_bets(dup) == pytest.approx(1.0, abs=0.15)
    ind = _factor_panel(n_assets=5, rho=0.0, n=6000)
    assert st.effective_number_of_bets(ind) == pytest.approx(5.0, rel=0.15)


def test_effective_bets_falls_as_correlation_rises():
    lo = st.effective_number_of_bets(_factor_panel(n_assets=8, rho=0.1, n=4000))
    hi = st.effective_number_of_bets(_factor_panel(n_assets=8, rho=0.8, n=4000))
    assert lo > hi


def test_greedy_order_beats_a_random_order_in_sample():
    r = _factor_panel(n_assets=6, rho=0.3, n=3000)
    order = st.greedy_order(r, cost_bps=0.0)
    assert sorted(order) == sorted(r.columns)
    g = st.ordered_curve(r, order, cost_bps=0.0)
    rnd = st.random_subset_curve(r, draws=60, cost_bps=0.0)
    assert (g["vol"] <= rnd["vol_mean"] + 1e-9).all()


def test_ordered_curve_reports_effective_bets(planted):
    prices, cash, _ = planted
    rets = st.excess_returns(prices, cash)
    order = list(rets.columns)
    g = st.ordered_curve(rets, order, cost_bps=0.0)
    assert g["enb"].iloc[-1] >= g["enb"].iloc[0]
    assert len(g) == len(order)


def test_excess_returns_subtract_the_cash_leg(planted):
    prices, cash, _ = planted
    ex = st.excess_returns(prices, cash)
    raw = prices.pct_change().reindex(ex.index)
    c = cash.reindex(ex.index).pct_change()
    assert np.allclose(ex.iloc[1:, 0], (raw.iloc[:, 0] - c).iloc[1:], equal_nan=True)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"vol_reduction_total": 0.45, "vol_k1": 0.18, "vol_kmax": 0.10, "n_universe": 12,
         "avg_corr": 0.30, "floor_vol": 0.095, "third_gain": 0.08, "last_gain": 0.005,
         "stop_5pct": 4, "stop_2pct": 6, "enb_full": 3.4, "tail_gain": 0.03,
         "greedy_vol_at_stop": 0.085, "vol_at_stop": 0.11, "dispersion_at_k5": 0.03}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(vol_reduction_total=0.10))["signal"] == "Weak"
    assert st.verdict(_headline(vol_reduction_total=0.01))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(last_gain=0.05))["trad"] == "Mirage"
    assert st.verdict(_headline(dispersion_at_k5=0.90))["trad"] == "Fragile"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(enb_full=2.7))
    assert "2.7 effective bets" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
