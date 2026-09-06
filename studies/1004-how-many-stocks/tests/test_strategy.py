"""Strategy tests for Study 1004 — how many stocks, under three different questions."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from howmany import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Drawing portfolios
# --------------------------------------------------------------------------- #
def test_a_one_stock_portfolio_is_a_stock():
    R = st.synthetic_cross_section(n_stocks=10, n_days=500)
    P = st.draw_portfolios(R, 1, n_draws=20)
    assert P.shape == (20, 500)
    for row in P:
        assert any(np.allclose(row, R[c].to_numpy()) for c in R.columns)


def test_the_whole_basket_gives_one_unique_portfolio():
    R = st.synthetic_cross_section(n_stocks=8, n_days=400)
    P = st.draw_portfolios(R, 8, n_draws=10)
    assert np.allclose(P, P[0])
    assert np.allclose(P[0], R.mean(axis=1).to_numpy())


def test_draw_portfolios_declines_when_asked_for_too_many():
    R = st.synthetic_cross_section(n_stocks=5, n_days=300)
    assert st.draw_portfolios(R, 10).size == 0


def test_draw_portfolios_never_repeats_a_name_within_a_draw():
    """Sampling without replacement — otherwise 'N stocks' does not mean N stocks."""
    R = st.synthetic_cross_section(n_stocks=20, n_days=300, avg_corr=0.0)
    P = st.draw_portfolios(R, 2, n_draws=200, seed=7)
    singles = R.to_numpy().T
    for row in P[:20]:
        assert not any(np.allclose(row, s) for s in singles)


def test_usable_panel_drops_short_histories():
    idx = pd.bdate_range("2015-01-01", periods=1500)
    px = pd.DataFrame({"LONG": np.linspace(100, 200, 1500),
                       "SHORT": [np.nan] * 1200 + list(np.linspace(10, 20, 300))},
                      index=idx)
    out = st.usable_panel(px, ("LONG", "SHORT"), min_obs=1000)
    assert list(out.columns) == ["LONG"]


# --------------------------------------------------------------------------- #
# The textbook curve
# --------------------------------------------------------------------------- #
def test_volatility_falls_with_the_number_of_stocks():
    R = st.synthetic_cross_section(n_stocks=60, n_days=3000, avg_corr=0.30)
    c = st.volatility_curve(R, sizes=(1, 5, 20, 50), n_draws=120)
    assert c["mean_vol"].is_monotonic_decreasing


def test_volatility_flattens_towards_the_common_factor():
    """The floor is the systematic risk, and it is reached, not approached forever."""
    R = st.synthetic_cross_section(n_stocks=60, n_days=4000, avg_corr=0.30,
                                   stock_vol=0.32)
    c = st.volatility_curve(R, sizes=(20, 40, 60), n_draws=120)
    floor = 0.32 * np.sqrt(0.30)
    assert abs(c.loc[60, "mean_vol"] - floor) < 0.03
    assert (c.loc[20, "mean_vol"] - c.loc[60, "mean_vol"]) < 0.03


def test_higher_correlation_means_a_higher_floor_and_earlier_flattening():
    lo = st.synthetic_cross_section(n_stocks=60, n_days=3000, avg_corr=0.10)
    hi = st.synthetic_cross_section(n_stocks=60, n_days=3000, avg_corr=0.50)
    cl = st.volatility_curve(lo, sizes=(1, 5, 10, 20, 40, 60), n_draws=120)
    ch = st.volatility_curve(hi, sizes=(1, 5, 10, 20, 40, 60), n_draws=120)
    assert ch.loc[60, "mean_vol"] > cl.loc[60, "mean_vol"]
    assert st.stocks_for_share(ch, "mean_vol", 0.90) <= \
        st.stocks_for_share(cl, "mean_vol", 0.90)


def test_the_textbook_number_is_reproduced_on_real_data():
    """The claim being criticised must first be shown to be true."""
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    c = st.volatility_curve(R, n_draws=200)
    assert st.stocks_for_share(c, "mean_vol", 0.90) <= 20


# --------------------------------------------------------------------------- #
# The curve that matters
# --------------------------------------------------------------------------- #
def test_terminal_wealth_dispersion_also_falls():
    R = st.synthetic_cross_section(n_stocks=60, n_days=3000, mu_dispersion=0.06)
    c = st.terminal_wealth_curve(R, sizes=(1, 5, 20, 50), n_draws=200)
    assert c["log_sd"].is_monotonic_decreasing


def test_terminal_wealth_needs_more_stocks_than_volatility_does():
    """The headline. Both curves on the same data, same draws, different answers."""
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    v = st.volatility_curve(R, n_draws=250)
    w = st.terminal_wealth_curve(R, n_draws=250)
    n_vol = st.stocks_for_share(v, "mean_vol", 0.90)
    n_wealth = st.stocks_for_share(w, "log_sd", 0.90)
    assert n_wealth > n_vol


def test_the_two_curves_are_driven_by_different_things():
    """The identification: one knob moves one curve and not the other.

    If the terminal-wealth curve were merely the volatility curve restated, dispersion in
    expected returns could not move them independently. It does.
    """
    base = st.synthetic_cross_section(n_stocks=60, n_days=4000, avg_corr=0.30,
                                      mu_dispersion=0.0)
    disp = st.synthetic_cross_section(n_stocks=60, n_days=4000, avg_corr=0.30,
                                      mu_dispersion=0.10)
    sizes = (1, 5, 10, 20, 40, 60)
    vb = st.volatility_curve(base, sizes=sizes, n_draws=150)
    vd = st.volatility_curve(disp, sizes=sizes, n_draws=150)
    wb = st.terminal_wealth_curve(base, sizes=sizes, n_draws=150)
    wd = st.terminal_wealth_curve(disp, sizes=sizes, n_draws=150)
    # Compared at N=20, not N=60: with 60 of 60 names there is exactly ONE possible
    # portfolio, so the dispersion across draws is identically zero in every world and the
    # comparison would be between two rounding errors.
    assert abs(vb.loc[20, "mean_vol"] - vd.loc[20, "mean_vol"]) < 0.02
    assert wd.loc[20, "log_sd"] > wb.loc[20, "log_sd"] * 1.5


def test_holding_the_whole_basket_leaves_no_dispersion_to_measure():
    """The boundary condition, and the reason the test above compares at N=20."""
    R = st.synthetic_cross_section(n_stocks=40, n_days=4000, mu_dispersion=0.08)
    c = st.terminal_wealth_curve(R, sizes=(40,), n_draws=100)
    assert c.loc[40, "log_sd"] == pytest.approx(0.0, abs=1e-9)
    partial = st.terminal_wealth_curve(R, sizes=(20,), n_draws=200)
    assert partial.loc[20, "log_sd"] > 0.01


def test_the_spread_of_outcomes_at_twenty_stocks_is_large():
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    c = st.terminal_wealth_curve(R, sizes=(20,), n_draws=400)
    assert c.loc[20, "ratio_95_05"] > 1.5


# --------------------------------------------------------------------------- #
# Tracking error
# --------------------------------------------------------------------------- #
def test_tracking_error_falls_with_holdings():
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    bench = px[data.MARKET].pct_change()
    c = st.tracking_error_curve(R, bench, sizes=(5, 20, 40), n_draws=150)
    assert c["mean_te"].is_monotonic_decreasing


def test_tracking_error_does_not_reach_zero_on_a_subset_of_the_index():
    """Forty names is not the S&P 500, so a floor remains — stated rather than hidden."""
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    bench = px[data.MARKET].pct_change()
    c = st.tracking_error_curve(R, bench, sizes=(40,), n_draws=100)
    assert c.loc[40, "mean_te"] > 0.02


# --------------------------------------------------------------------------- #
# Marginal benefit and the summary statistic
# --------------------------------------------------------------------------- #
def test_share_of_benefit_runs_from_zero_to_one():
    R = st.synthetic_cross_section(n_stocks=40, n_days=2000)
    m = st.marginal_benefit(st.volatility_curve(R, sizes=(1, 10, 40), n_draws=80),
                            "mean_vol")
    assert m["share_of_benefit"].iloc[0] == pytest.approx(0.0)
    assert m["share_of_benefit"].iloc[-1] == pytest.approx(1.0)


def test_stocks_for_share_is_monotone_in_the_share_asked_for():
    R = st.synthetic_cross_section(n_stocks=60, n_days=3000)
    c = st.volatility_curve(R, n_draws=120)
    a = st.stocks_for_share(c, "mean_vol", 0.50)
    b = st.stocks_for_share(c, "mean_vol", 0.90)
    d = st.stocks_for_share(c, "mean_vol", 0.99)
    assert a <= b <= d


def test_marginal_benefit_handles_a_degenerate_curve():
    assert st.marginal_benefit(pd.DataFrame({"mean_vol": [0.2]}, index=[1]),
                               "mean_vol").empty
    assert np.isnan(st.stocks_for_share(pd.DataFrame({"v": [0.2]}, index=[1]), "v"))


# --------------------------------------------------------------------------- #
# The mechanism: skew
# --------------------------------------------------------------------------- #
def test_the_median_portfolio_lags_the_mean_and_catches_up_slowly():
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    s = st.skew_and_the_median_portfolio(R, sizes=(1, 5, 20, 40), n_draws=300)
    assert (s["shortfall"] > 0).all()
    assert s["shortfall"].is_monotonic_decreasing


def test_returns_are_concentrated_in_a_few_names():
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    c = st.concentration_of_returns(R)
    assert c["share_from_top_10pct"] > 0.20
    assert c["mean_stock_return"] > c["median_stock_return"]


def test_a_symmetric_cross_section_has_no_median_shortfall():
    """The control: kill the skew and the mechanism disappears."""
    R = st.synthetic_cross_section(n_stocks=40, n_days=4000, mu_dispersion=0.0)
    s = st.skew_and_the_median_portfolio(R, sizes=(5, 20), n_draws=300)
    assert abs(s.loc[5, "shortfall"]) < 0.10


def test_buy_and_hold_disperses_more_than_rebalancing():
    px = data.load_prices()
    R = st.usable_panel(px, data.NAMES)
    d = st.rebalanced_vs_held(R, 20, n_draws=150)
    assert d["held_log_sd"] > d["rebalanced_log_sd"]


def test_rebalanced_vs_held_declines_on_too_small_a_basket():
    R = st.synthetic_cross_section(n_stocks=5, n_days=500)
    assert st.rebalanced_vs_held(R, 20) == {}


# --------------------------------------------------------------------------- #
# The synthetic cross-section
# --------------------------------------------------------------------------- #
def test_the_synthetic_correlation_is_what_it_claims():
    R = st.synthetic_cross_section(n_stocks=40, n_days=20000, avg_corr=0.35)
    c = R.corr().to_numpy()
    off = c[~np.eye(len(c), dtype=bool)]
    assert off.mean() == pytest.approx(0.35, abs=0.03)


def test_the_synthetic_volatility_is_what_it_claims():
    R = st.synthetic_cross_section(n_stocks=20, n_days=20000, stock_vol=0.32)
    assert (R.std(ddof=1) * np.sqrt(252)).mean() == pytest.approx(0.32, rel=0.05)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_available": 40, "years": 25.0, "vol_at_1": 0.331, "vol_at_max": 0.166,
         "n_for_90_vol": 14.0, "n_for_90_wealth": 31.0, "ratio_at_vol_n": 1.94,
         "share_negative": 0.05, "share_from_top_10pct": 0.34,
         "shortfall_at_vol_n": 0.11, "shortfall_at_wealth_n": 0.06}
    h.update(over)
    return h


def test_verdict_signal_confirms_the_textbook_curve_first():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(n_for_90_vol=30))["signal"] == "Partial"
    assert st.verdict(_headline(n_for_90_vol=45))["signal"] == "Busted"


def test_verdict_tradability_keys_off_the_ratio_between_the_two_answers():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(n_for_90_wealth=20))["trad"] == "Partial"
    assert st.verdict(_headline(n_for_90_wealth=15))["trad"] == "Mirage"


def test_verdict_prose_grants_the_textbook_its_point_before_taking_it_apart():
    v = st.verdict(_headline())
    assert "is real" in v["signal_why"]
    assert "not making an arithmetic error" in v["signal_why"]
    assert "wrong question" in v["trad_why"]
    assert "skew, not covariance" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
