"""Strategy tests for Study 1006 — the median stock, the index, and variance drag."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from moststocks import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def test_variance_drag_is_half_the_variance():
    assert st.variance_drag(0.10, 0.20) == pytest.approx(0.02)
    assert st.variance_drag(0.10, 0.40) == pytest.approx(0.08)


def test_drag_quadruples_when_volatility_doubles():
    assert st.variance_drag(0, 0.60) == pytest.approx(4 * st.variance_drag(0, 0.30))


def test_a_riskless_series_has_no_drag():
    """Compared in LOG space, which is the only place the identity holds."""
    r = pd.Series([0.0004] * 2520, index=pd.bdate_range("2010-01-01", periods=2520))
    assert st.arithmetic_mean(r) - st.log_growth(r) == pytest.approx(0.0, abs=1e-3)


def test_comparing_against_the_compounded_return_would_get_the_sign_wrong():
    """The trap this module avoids, pinned so it cannot come back.

    For a constant positive return the *compounded* annual figure exceeds 252× the daily one,
    so 'arithmetic minus geometric' is NEGATIVE on a riskless series — which would make the
    drag look like a bonus. Ordinary compounding is not variance drag.
    """
    r = pd.Series([0.0004] * 2520, index=pd.bdate_range("2010-01-01", periods=2520))
    assert st.geometric_mean(r) > st.arithmetic_mean(r)
    assert st.arithmetic_mean(r) - st.geometric_mean(r) < 0


def test_arithmetic_exceeds_log_growth_whenever_there_is_variance():
    rng = np.random.default_rng(1006)
    r = pd.Series(rng.normal(0.0004, 0.02, 5000),
                  index=pd.bdate_range("2000-01-01", periods=5000))
    assert st.arithmetic_mean(r) > st.log_growth(r)


# --------------------------------------------------------------------------- #
# The breakeven condition — the study's deliverable
# --------------------------------------------------------------------------- #
def test_breakeven_volatility_matches_its_algebra():
    assert st.breakeven_volatility(0.08, 0.0) == pytest.approx(np.sqrt(0.16))
    assert st.breakeven_volatility(0.10, 0.02) == pytest.approx(np.sqrt(0.16))
    assert st.breakeven_volatility(0.02, 0.05) == 0.0       # no excess drift at all


def test_at_the_breakeven_volatility_the_median_stops_winning():
    """The condition, checked by simulation rather than trusted."""
    drift, cash = 0.10, 0.02
    sigma = st.breakeven_volatility(drift, cash)
    for vol, expect_win in ((sigma * 0.6, True), (sigma * 1.4, False)):
        R = st.synthetic_cross_section(n_stocks=400, n_days=6000, drift=drift, vol=vol)
        finals = np.expm1(np.log1p(R).sum())
        years = len(R) / 252
        cash_total = np.expm1(np.log1p(cash) * years)
        assert bool(finals.median() > cash_total) is expect_win


def test_these_large_caps_sit_well_below_their_threshold():
    """Which is exactly why the headline fails on them."""
    px = data.load_prices()
    R = _panel(px)
    c = st.median_beats_cash_condition(R, _cash(px, R.index))
    assert c["mean_vol"] < c["mean_breakeven"]
    assert c["median_headroom"] > 0
    assert c["share_above_threshold"] < 0.35


def test_the_measured_drag_matches_the_formula():
    """The identification. If these disagree, nothing downstream means anything."""
    R = st.synthetic_cross_section(n_stocks=50, n_days=20000, vol=0.35)
    d = st.drag_table(R)
    err = (d["measured_drag"] - d["predicted_drag"]).abs()
    assert err.median() < 0.01


def test_the_drag_formula_holds_on_real_stocks_too():
    px = data.load_prices()
    R = _panel(px)
    d = st.drag_table(R)
    corr = d["measured_drag"].corr(d["predicted_drag"])
    assert corr > 0.85


def test_means_decline_on_a_short_series():
    assert np.isnan(st.geometric_mean(pd.Series([0.01])))
    assert np.isnan(st.arithmetic_mean(pd.Series([0.01])))


# --------------------------------------------------------------------------- #
# The distribution of outcomes
# --------------------------------------------------------------------------- #
def test_holding_outcomes_compounds_correctly():
    idx = pd.bdate_range("2015-01-01", periods=600)
    R = pd.DataFrame({"A": np.full(600, 0.001)}, index=idx)
    cash = pd.Series(np.zeros(600), index=idx)
    out = st.holding_outcomes(R, cash, 252, step=252)
    assert out["stock"].iloc[0] == pytest.approx(1.001 ** 252 - 1, rel=1e-9)
    assert out["cash"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_beating_cash_is_computed_against_cash_not_zero():
    idx = pd.bdate_range("2015-01-01", periods=600)
    R = pd.DataFrame({"A": np.full(600, 0.0001)}, index=idx)
    cash = pd.Series(np.full(600, 0.0002), index=idx)
    out = st.holding_outcomes(R, cash, 252, step=252)
    assert out["stock"].iloc[0] > 0            # made money
    assert not out["beat_cash"].iloc[0]        # and still lost to cash


def test_outcome_distribution_handles_an_empty_frame():
    assert st.outcome_distribution(pd.DataFrame()) == {}


def test_the_share_beating_cash_falls_with_horizon():
    """The counterintuitive core, on synthetic data where the drift is known and positive."""
    R = st.synthetic_cross_section(n_stocks=120, n_days=8000, drift=0.08, vol=0.45)
    cash = pd.Series(np.full(len(R), 0.02 / 252), index=R.index)
    sw = st.horizon_sweep(R, cash, horizons_years=(1, 5, 15), step=126)
    assert sw["share_beat_cash"].iloc[-1] < sw["share_beat_cash"].iloc[0]


def test_the_average_return_rises_with_horizon_at_the_same_time():
    """Both at once, which is what makes it a paradox rather than a triviality."""
    R = st.synthetic_cross_section(n_stocks=120, n_days=8000, drift=0.08, vol=0.45)
    cash = pd.Series(np.full(len(R), 0.02 / 252), index=R.index)
    sw = st.horizon_sweep(R, cash, horizons_years=(1, 5, 15), step=126)
    assert sw["mean_return"].is_monotonic_increasing
    assert sw["share_beat_cash"].iloc[-1] < sw["share_beat_cash"].iloc[0]


def test_with_no_volatility_the_share_beating_cash_does_not_fall():
    """The control: turn off the mechanism and the effect must vanish."""
    R = st.synthetic_cross_section(n_stocks=60, n_days=6000, drift=0.08, vol=0.01)
    cash = pd.Series(np.full(len(R), 0.02 / 252), index=R.index)
    sw = st.horizon_sweep(R, cash, horizons_years=(1, 10), step=126)
    assert sw["share_beat_cash"].min() > 0.95


def test_higher_volatility_means_fewer_winners_at_the_same_drift():
    cashless = lambda R: pd.Series(np.full(len(R), 0.02 / 252), index=R.index)
    lo = st.synthetic_cross_section(n_stocks=100, n_days=6000, drift=0.08, vol=0.20)
    hi = st.synthetic_cross_section(n_stocks=100, n_days=6000, drift=0.08, vol=0.60)
    a = st.horizon_sweep(lo, cashless(lo), horizons_years=(10,), step=126)
    b = st.horizon_sweep(hi, cashless(hi), horizons_years=(10,), step=126)
    assert b.loc[10, "share_beat_cash"] < a.loc[10, "share_beat_cash"]


def test_the_headline_FAILS_on_surviving_large_caps():
    """Pre-registered as a confirmation; the data said the opposite, decisively.

    The share of holdings beating bills RISES with horizon on this basket, from roughly
    two-thirds at one year to nearly all at fifteen. Bessembinder's result is not a universal
    property of equities — it is what happens to a cross-section whose volatility exceeds
    sqrt(2·(drift − cash)), and surviving large caps do not.
    """
    px = data.load_prices()
    R = _panel(px)
    sw = st.horizon_sweep(R, _cash(px, R.index), horizons_years=(1, 10, 15), step=126)
    assert sw.loc[15, "share_beat_cash"] > sw.loc[1, "share_beat_cash"]
    assert sw.loc[15, "share_beat_cash"] > 0.80


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def test_wealth_creation_is_concentrated_even_where_the_headline_fails():
    """The half of Bessembinder's result that DOES survive on this basket."""
    px = data.load_prices()
    R = _panel(px)
    c = st.wealth_concentration(R, _cash(px, R.index))
    assert c["top_10pct_share"] > 0.25
    assert c["top_50pct_share"] > 0.80
    # ...while the other half does not: essentially all of these survivors beat cash
    assert c["share_beat_cash"] > 0.80


def test_the_concentration_shares_are_ordered():
    px = data.load_prices()
    R = _panel(px)
    c = st.wealth_concentration(R, _cash(px, R.index))
    shares = [c[f"top_{p}pct_share"] for p in (2, 5, 10, 25, 50)]
    assert shares == sorted(shares)


def test_concentration_is_measured_on_dollars_not_percentages():
    """A doubling from $1 and a doubling from $60 are not the same contribution."""
    idx = pd.bdate_range("2000-01-01", periods=2520)
    R = pd.DataFrame({
        "BIG": np.full(2520, 0.002),      # compounds to a large number
        "SMALL": np.full(2520, 0.0001),
    }, index=idx)
    c = st.wealth_concentration(R, pd.Series(np.zeros(2520), index=idx))
    assert c["best_name"] == "BIG"
    assert c["top_50pct_share"] > 0.9


def test_wealth_concentration_declines_on_an_empty_panel():
    idx = pd.bdate_range("2020-01-01", periods=50)
    assert st.wealth_concentration(pd.DataFrame({"A": np.zeros(50)}, index=idx),
                                   pd.Series(np.zeros(50), index=idx)) == {}


# --------------------------------------------------------------------------- #
# The reconciliation
# --------------------------------------------------------------------------- #
def test_the_rebalanced_basket_beats_its_own_median_member():
    """The paradox, resolved and asserted."""
    px = data.load_prices()
    R = _panel(px)
    rec = st.index_reconciliation(R, _cash(px, R.index))
    assert rec["rebalanced_equal_weight"] > rec["median_single"]


def test_a_portfolio_is_far_less_volatile_than_its_members():
    px = data.load_prices()
    R = _panel(px)
    rec = st.index_reconciliation(R, _cash(px, R.index))
    assert rec["portfolio_vol"] < rec["single_vol"] / 1.5
    assert rec["drag_saved"] > 0.01


def test_the_drag_saved_is_what_explains_the_gap():
    """Quantitative, not just directional: the saving should be material over the sample."""
    px = data.load_prices()
    R = _panel(px)
    rec = st.index_reconciliation(R, _cash(px, R.index))
    years = len(R.dropna()) / 252
    implied = np.expm1(np.log1p(rec["drag_saved"]) * years)
    assert implied > 0.20                       # worth more than 20% cumulatively


def test_buy_and_hold_becomes_concentrated():
    px = data.load_prices()
    R = _panel(px)
    bh = st.buy_and_hold_index(R)
    assert bh["max_weight"].iloc[-1] > bh["max_weight"].iloc[0] * 2
    assert bh["effective_n"].iloc[-1] < bh["effective_n"].iloc[0]


def test_buy_and_hold_starts_equally_weighted():
    px = data.load_prices()
    R = _panel(px)
    bh = st.buy_and_hold_index(R)
    assert bh["effective_n"].iloc[0] == pytest.approx(R.shape[1], rel=0.05)


def test_buy_and_hold_index_handles_an_empty_frame():
    assert st.buy_and_hold_index(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# What it means for a concentrated book
# --------------------------------------------------------------------------- #
def test_concentrated_portfolios_beat_the_index_less_often():
    px = data.load_prices()
    R = _panel(px)
    o = st.concentrated_portfolio_odds(R, _cash(px, R.index), sizes=(1, 5, 20),
                                       horizon_years=10, n_draws=300)
    assert o.loc[1, "share_beat_index"] < o.loc[20, "share_beat_index"]


def test_a_concentrated_portfolio_has_a_wider_spread_of_outcomes():
    px = data.load_prices()
    R = _panel(px)
    o = st.concentrated_portfolio_odds(R, _cash(px, R.index), sizes=(1, 20),
                                       horizon_years=10, n_draws=300)
    assert (o.loc[1, "p90"] - o.loc[1, "p10"]) > (o.loc[20, "p90"] - o.loc[20, "p10"])


def test_concentration_odds_decline_when_the_horizon_exceeds_the_data():
    px = data.load_prices()
    R = _panel(px)
    assert st.concentrated_portfolio_odds(R, _cash(px, R.index),
                                          horizon_years=500).empty


# --------------------------------------------------------------------------- #
# Survivorship, measured
# --------------------------------------------------------------------------- #
def test_survivorship_bias_runs_against_the_finding():
    """The comfortable direction — and quantified rather than asserted."""
    s = st.survivorship_experiment(n_stocks=400, n_days=6000, vol=0.45)
    assert s["delist_rate"] > 0.0
    assert s["median_survivors"] > s["median_all"]
    assert s["bias"] > 0


def test_more_volatility_means_more_delisting_and_more_bias():
    lo = st.survivorship_experiment(n_stocks=400, n_days=6000, vol=0.30)
    hi = st.survivorship_experiment(n_stocks=400, n_days=6000, vol=0.60)
    assert hi["delist_rate"] > lo["delist_rate"]


# --------------------------------------------------------------------------- #
# The synthetic cross-section
# --------------------------------------------------------------------------- #
def test_every_synthetic_stock_has_the_same_expected_return():
    """Dispersion in outcomes must be pure luck, or the control proves nothing.

    Scored against the standard error of a realised mean, not against zero: with σ = 35% and
    79 years the realised annual mean of a single name has an SE near 4pp, so demanding that
    the cross-section of *realised* means be tight would be demanding the impossible. What
    must hold is that the spread matches sampling error and no more.
    """
    n_days = 20000
    R = st.synthetic_cross_section(n_stocks=200, n_days=n_days, drift=0.08, vol=0.35)
    ann = R.mean() * 252
    assert ann.mean() == pytest.approx(0.08, abs=0.01)
    expected_se = 0.35 / np.sqrt(n_days / 252)
    assert ann.std(ddof=1) == pytest.approx(expected_se, rel=0.25)


def test_the_synthetic_median_falls_further_below_the_mean_as_volatility_rises():
    """Measured in log space, where the gap is σ²/2 × years and therefore monotone.

    In levels the comparison is dominated by a handful of enormous draws and is not reliably
    ordered — at 60% volatility the median has collapsed toward zero, so mean-minus-median
    stops being a sensible scale.
    """
    gaps = []
    for vol in (0.10, 0.35, 0.60):
        R = st.synthetic_cross_section(n_stocks=400, n_days=6000, drift=0.08, vol=vol)
        lg = np.log1p(R).sum()
        gaps.append(float(R.mean().mean() * len(R) - lg.mean()))
    assert gaps[0] < gaps[1] < gaps[2]


def test_the_synthetic_log_gap_matches_the_theory():
    years = 6000 / 252
    for vol in (0.20, 0.40):
        R = st.synthetic_cross_section(n_stocks=400, n_days=6000, drift=0.08, vol=vol)
        measured = float(R.mean().mean() * 252 - np.log1p(R).mean().mean() * 252)
        assert measured == pytest.approx(st.variance_drag(0.08, vol), rel=0.15)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _panel(px):
    """Each name over ITS OWN history, not the common intersection.

    Intersecting fifty tapes throws away two decades — the shortest name starts in 2010 — and
    the whole question is about long holding periods. Bessembinder scores each stock over its
    own lifetime, and so does this.
    """
    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2000]
    return px[cols].pct_change().dropna(how="all")


def _cash(px, idx):
    if data.BILLS in px.columns:
        return px[data.BILLS].pct_change().reindex(idx).fillna(0.0)
    return pd.Series(np.full(len(idx), 0.02 / 252), index=idx)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_names": 50, "share_beat_cash_long": 0.91, "share_beat_cash_short": 0.68,
         "long_years": 10.0, "mean_drag": 0.052, "predicted_drag": 0.055,
         "drag_corr": 0.97, "mean_drift": 0.152, "cash_log_growth": 0.014,
         "mean_breakeven": 0.525, "mean_vol": 0.322, "median_headroom": 0.19,
         "share_above_threshold": 0.06,
         "top_decile_pct": 10.0, "top_10pct_share": 0.41, "share_lost_money": 0.02,
         "surv_bias": 0.34, "median_single": 6.2, "cash": 1.5, "rebalanced": 11.8,
         "single_vol": 0.32, "portfolio_vol": 0.17, "drag_saved": 0.036,
         "odds_horizon": 10.0, "odds_index_5": 0.31, "odds_index_20": 0.44,
         "odds_index_50": 0.50}
    h.update(over)
    return h


def test_verdict_signal_reports_the_failure_on_this_basket():
    assert st.verdict(_headline())["signal"] == "Busted"
    assert st.verdict(_headline(share_beat_cash_long=0.55))["signal"] == "Partial"
    assert st.verdict(_headline(share_beat_cash_long=0.44))["signal"] == "Confirmed"


def test_verdict_tradability_keys_off_the_concentration_penalty():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(odds_index_5=0.45))["trad"] == "Partial"
    assert st.verdict(_headline(odds_index_5=0.49))["trad"] == "Mirage"


def test_verdict_prose_states_the_failure_and_the_condition():
    v = st.verdict(_headline())
    assert "Not on this basket" in v["signal_why"]
    assert "all survived" in v["signal_why"]
    assert "sqrt(2" in v["signal_why"]
    assert "not a universal" in v["signal_why"]
    assert "don't buy few stocks" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
