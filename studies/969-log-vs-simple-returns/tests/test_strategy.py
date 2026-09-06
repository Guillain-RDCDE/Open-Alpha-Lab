"""Strategy tests for Study 969 — arithmetic identities, which are the strongest tests there are."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from log_vs_simple import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The identities
# --------------------------------------------------------------------------- #
def test_conversions_are_exact_inverses():
    R = np.array([-0.5, -0.01, 0.0, 0.01, 0.5, 3.0])
    assert np.allclose(st.convert_log_to_simple(st.convert_simple_to_log(R)), R)


def test_log_returns_sum_to_the_total_log_return(planted):
    prices, _, _ = planted
    p = prices.iloc[:, 0]
    r = st.log_returns(p).dropna()
    assert r.sum() == pytest.approx(np.log(p.iloc[-1] / p.iloc[0]))


def test_simple_returns_compound_to_the_total(planted):
    prices, _, _ = planted
    p = prices.iloc[:, 0]
    R = st.simple_returns(p).dropna()
    assert (1 + R).prod() == pytest.approx(p.iloc[-1] / p.iloc[0])


def test_simple_returns_are_linear_across_assets(planted):
    """The identity that decides the portfolio rule."""
    prices, _, _ = planted
    R = st.simple_returns(prices).dropna()
    w = np.full(prices.shape[1], 1 / prices.shape[1])
    port = st.portfolio_simple(R, w)
    manual = R.mul(w, axis=1).sum(axis=1)
    assert np.allclose(port.to_numpy(), manual.to_numpy())


def test_log_returns_are_not_linear_across_assets(planted):
    prices, _, _ = planted
    R = st.simple_returns(prices).dropna()
    L = st.log_returns(prices).dropna()
    w = np.full(prices.shape[1], 1 / prices.shape[1])
    right = st.portfolio_simple(R, w)
    wrong = np.expm1(st.portfolio_from_logs(L, w))
    assert not np.allclose(right.to_numpy(), wrong.to_numpy())


def test_the_log_weighting_mistake_always_understates(planted):
    """Jensen's inequality points one way, so this is an inequality, not a tendency."""
    prices, _, _ = planted
    R = st.simple_returns(prices).dropna()
    L = st.log_returns(prices).dropna()
    w = np.full(prices.shape[1], 1 / prices.shape[1])
    right = st.portfolio_simple(R, w)
    wrong = np.expm1(st.portfolio_from_logs(L, w))
    assert (right >= wrong - 1e-12).all()


# --------------------------------------------------------------------------- #
# Volatility drag
# --------------------------------------------------------------------------- #
def test_drag_matches_half_variance_at_low_volatility():
    idx = pd.bdate_range("2000-01-03", periods=4000)
    rng = np.random.default_rng(969)
    R = pd.Series(rng.normal(0.0002, 0.004, 4000), index=idx)   # ~6% annualised vol
    p = pd.DataFrame({"X": 100 * (1 + R).cumprod()})
    t = st.drag_table(p).loc["X"]
    assert t["residual_ann"] == pytest.approx(0.0, abs=0.002)
    assert t["gap_ann"] == pytest.approx(t["half_var_ann"], rel=0.2)


def test_drag_grows_with_the_square_of_volatility():
    rng = np.random.default_rng(969)
    gaps = {}
    for vol in (0.10, 0.20, 0.40):
        R = pd.Series(rng.normal(0.0003, vol / np.sqrt(252), 6000),
                      index=pd.bdate_range("2000-01-03", periods=6000))
        p = pd.DataFrame({"X": 100 * (1 + R).cumprod()})
        gaps[vol] = st.drag_table(p).loc["X", "gap_ann"]
    assert gaps[0.40] > gaps[0.20] > gaps[0.10] > 0
    assert gaps[0.40] / gaps[0.20] == pytest.approx(4.0, rel=0.35)


def test_cagr_from_means_matches_the_realised_cagr_at_moderate_vol():
    rng = np.random.default_rng(969)
    R = pd.Series(rng.normal(0.0004, 0.01, 8000))
    p = 100 * (1 + R).cumprod()
    approx = st.cagr_from_means(float(R.mean()), float(R.var(ddof=1)))
    realised = float(p.iloc[-1] / p.iloc[0]) ** (252 / len(R)) - 1
    assert approx == pytest.approx(realised, rel=0.05)


def test_geometric_mean_never_exceeds_arithmetic(planted):
    prices, _, _ = planted
    t = st.drag_table(prices)
    assert (t["mean_log_ann"] <= t["mean_simple_ann"] + 1e-12).all()
    assert (t["gap_ann"] >= 0).all()


# --------------------------------------------------------------------------- #
# The portfolio and the statistics
# --------------------------------------------------------------------------- #
def test_portfolio_error_is_the_rebalancing_bonus(planted):
    """The 'error' is exactly the excess growth rate a rebalanced book earns."""
    prices, _, _ = planted
    tk = tuple(prices.columns[:3])
    err = st.portfolio_error(prices, tk)
    bonus = st.rebalancing_bonus(prices, tk)
    assert err["cagr_gap"] > 0
    # Against the weighted GEOMETRIC benchmark the identity is near-exact; against the
    # arithmetic one it is not, and conflating the two is the usual confusion.
    assert err["cagr_gap"] == pytest.approx(bonus["bonus"], abs=0.002)
    assert bonus["bonus"] != pytest.approx(bonus["bonus_vs_arithmetic"], abs=1e-6)


def test_portfolio_error_vanishes_for_a_single_asset(planted):
    prices, _, _ = planted
    err = st.portfolio_error(prices, (prices.columns[0],))
    assert err["cagr_gap"] == pytest.approx(0.0, abs=1e-9)


def test_sharpe_gap_is_positive_and_grows_with_volatility():
    rng = np.random.default_rng(969)
    cols = {}
    for name, vol in (("calm", 0.08), ("wild", 0.60)):
        R = rng.normal(0.0004, vol / np.sqrt(252), 6000)
        cols[name] = 100 * np.cumprod(1 + R)
    p = pd.DataFrame(cols, index=pd.bdate_range("2000-01-03", periods=6000))
    g = st.sharpe_gap(p)
    assert g.loc["wild", "gap"] > g.loc["calm", "gap"]
    assert abs(g.loc["calm", "gap"]) < 0.05


def test_beta_is_barely_affected_by_the_convention(planted):
    prices, _, _ = planted
    b = st.beta_gap(prices, prices.columns[0], prices.columns[1])
    assert abs(b["relative_gap"]) < 0.10


def test_annualisation_table_orders_the_four_methods(planted):
    prices, _, _ = planted
    R = st.simple_returns(prices.iloc[:, 0]).dropna()
    t = st.annualisation_table(R)
    assert len(t) == 4
    # The ordering that always holds is between the two COMPOUNDED numbers: the geometric
    # mean can never exceed the arithmetic one. (The un-compounded "arithmetic x 252" is not
    # comparable to either — it is a different quantity, which is the point of the table.)
    assert (t.loc["exp(mean log x 252) - 1", "value"]
            <= t.loc["(1 + mean)^252 - 1", "value"] + 1e-9)
    assert t.loc["exp(mean log x 252) - 1", "value"] == pytest.approx(
        t.loc["exp((mu - var/2) x 252) - 1", "value"], rel=0.35)


def test_drag_curve_is_quadratic():
    c = st.drag_curve(vols=np.array([0.0, 0.2, 0.4]), mu_ann=0.08)
    assert c.loc[0.0, "geometric_approx"] == pytest.approx(0.08)
    assert c.loc[0.4, "geometric_approx"] == pytest.approx(0.08 - 0.08)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"max_gap_ann": 0.30, "calm_ticker": "BIL", "calm_vol": 0.003, "calm_gap": 0.00001,
         "wild_ticker": "BTC-USD", "wild_vol": 0.65, "half_var_explains": 0.9,
         "max_sharpe_gap": 0.25, "understates_always": True, "portfolio_n": 8,
         "portfolio_cagr_gap": 0.02, "portfolio_terminal_ratio": 1.4, "portfolio_years": 12.0}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(max_gap_ann=0.02))["signal"] == "Weak"
    assert st.verdict(_headline(max_gap_ann=0.001, understates_always=False))["signal"] == "None"


def test_verdict_usefulness_follows_the_direction_of_the_error():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(understates_always=False))["trad"] == "Fragile"
    assert st.verdict(_headline(understates_always=False,
                                max_gap_ann=0.001))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(max_gap_ann=0.37))
    assert "37" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
