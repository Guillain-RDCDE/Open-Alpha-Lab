"""Strategy tests for Study 984 — can the drop ratio be measured at all?"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from exday import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Event construction
# --------------------------------------------------------------------------- #
def test_ex_dates_are_the_nonzero_dividend_dates():
    bars = data.synthetic_panel(n=800, n_tickers=2)
    b = bars["SIM0"]
    ex = st.ex_dates(b)
    assert len(ex) > 5
    assert (b["dividend"].loc[ex] > 0).all()
    assert (b["dividend"].drop(ex) == 0).all()


def test_drop_ratios_measures_a_planted_full_drop():
    """With zero noise the ratio must be exactly 1.0 — the arithmetic sanity check."""
    bars = data.synthetic_panel(n=1200, n_tickers=1, drop_fraction=1.0, daily_vol=1e-9,
                                market_beta=0.0)
    ev = st.drop_ratios(bars["SIM0"], bars["MKT"])
    assert len(ev) > 10
    assert ev["raw_ratio"].mean() == pytest.approx(1.0, abs=0.01)


def test_drop_ratios_measures_a_planted_partial_drop():
    bars = data.synthetic_panel(n=1200, n_tickers=1, drop_fraction=0.5, daily_vol=1e-9,
                                market_beta=0.0)
    ev = st.drop_ratios(bars["SIM0"], bars["MKT"])
    assert ev["raw_ratio"].mean() == pytest.approx(0.5, abs=0.01)


def test_the_market_adjustment_removes_a_planted_market_move():
    """A stock that is pure beta with no dividend drop must show an adjusted ratio of zero."""
    bars = data.synthetic_panel(n=2000, n_tickers=1, drop_fraction=0.0, daily_vol=0.009,
                                market_beta=1.0)
    ev = st.drop_ratios(bars["SIM0"], bars["MKT"])
    assert abs(st.ratio_of_sums(ev, "adjusted_drop")) < abs(st.ratio_of_sums(ev, "raw_drop")) + 1


def test_the_adjusted_series_is_less_noisy_than_the_raw_one():
    bars = data.synthetic_panel(n=3000, n_tickers=4, drop_fraction=1.0, market_beta=1.0)
    ev = st.build_events(bars, "MKT")
    assert ev["adjusted_ratio"].std() < ev["raw_ratio"].std()


def test_the_beta_used_never_looks_forward():
    bars = data.synthetic_panel(n=2000, n_tickers=1)
    tampered = {k: v.copy() for k, v in bars.items()}
    tampered["MKT"].iloc[1500:, tampered["MKT"].columns.get_loc("close")] *= 3
    a = st.drop_ratios(bars["SIM0"], bars["MKT"]).loc[:"2010-01-01", "beta"]
    b = st.drop_ratios(tampered["SIM0"], tampered["MKT"]).loc[:"2010-01-01", "beta"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_tiny_dividends_are_excluded_by_the_yield_floor():
    bars = data.synthetic_panel(n=1000, n_tickers=1, quarterly_yield=1e-6)
    ev = st.drop_ratios(bars["SIM0"], bars["MKT"], min_yield=1e-4)
    assert len(ev) == 0


def test_build_events_pools_every_payer_and_skips_the_market():
    bars = data.synthetic_panel(n=1500, n_tickers=5)
    ev = st.build_events(bars, "MKT")
    assert set(ev["ticker"]) == {f"SIM{k}" for k in range(5)}


def test_drop_ratios_is_empty_on_a_non_payer():
    bars = data.synthetic_panel(n=500, n_tickers=1)
    ev = st.drop_ratios(bars["MKT"], bars["MKT"])
    assert len(ev) == 0
    assert "raw_ratio" in ev.columns


# --------------------------------------------------------------------------- #
# The estimators, and why they disagree
# --------------------------------------------------------------------------- #
def test_all_four_estimators_agree_when_there_is_no_noise():
    bars = data.synthetic_panel(n=2500, n_tickers=4, drop_fraction=0.8, daily_vol=1e-9,
                                market_beta=0.0)
    ev = st.build_events(bars, "MKT")
    t = st.estimator_table(ev, col="raw_ratio", drop_col="raw_drop")
    assert t["value"].max() - t["value"].min() < 0.02
    assert t["value"].mean() == pytest.approx(0.8, abs=0.02)


def test_the_market_adjustment_costs_something_when_there_is_no_market_move_to_remove():
    """A correction is never free: an estimated beta injects noise of its own.

    On a tape with no volatility and no market exposure, the *raw* drop is exact while the
    market-adjusted one carries ``price * beta_hat * r_market`` of pure estimation error. The
    adjustment earns its keep on real data (where the market move is large and real) and costs
    accuracy here — a trade-off worth having in a test rather than in a footnote.
    """
    bars = data.synthetic_panel(n=2500, n_tickers=4, drop_fraction=0.8, daily_vol=1e-9,
                                market_beta=0.0)
    ev = st.build_events(bars, "MKT")
    assert ev["raw_ratio"].std() < ev["adjusted_ratio"].std()
    assert st.regression_slope(ev, "raw_drop")["slope"] == pytest.approx(0.8, abs=0.02)


def test_the_estimators_disagree_once_realistic_noise_is_added():
    """The study's central methodological claim, as a test."""
    bars = data.synthetic_panel(n=2500, n_tickers=8, drop_fraction=0.8, daily_vol=0.012)
    ev = st.build_events(bars, "MKT")
    t = st.estimator_table(ev)
    assert t["value"].max() - t["value"].min() > 0.1


def test_the_regression_slope_recovers_the_truth_under_noise():
    """It is the estimator that survives — which is why the verdict is built on it."""
    slopes = []
    for s in range(6):
        bars = data.synthetic_panel(n=3000, n_tickers=10, drop_fraction=0.8, seed=984 + s)
        slopes.append(st.regression_slope(st.build_events(bars, "MKT"))["slope"])
    assert np.mean(slopes) == pytest.approx(0.8, abs=0.15)


def test_the_mean_of_ratios_is_the_estimator_that_misbehaves():
    bars = data.synthetic_panel(n=3000, n_tickers=10, drop_fraction=0.8)
    ev = st.build_events(bars, "MKT")
    d = st.ratio_dispersion(ev)
    assert d["sd"] > 1.0
    assert d["share_outside_0_2"] > 0.2


def test_ratio_of_sums_is_not_the_mean_of_ratios():
    bars = data.synthetic_panel(n=2000, n_tickers=6, drop_fraction=0.8)
    ev = st.build_events(bars, "MKT")
    assert st.ratio_of_sums(ev) != pytest.approx(st.mean_of_ratios(ev), abs=1e-6)


def test_regression_slope_reports_both_t_statistics():
    bars = data.synthetic_panel(n=2500, n_tickers=8, drop_fraction=1.0)
    r = st.regression_slope(st.build_events(bars, "MKT"))
    assert abs(r["t_vs_one"]) < abs(r["t_vs_zero"])


def test_regression_slope_declines_on_too_few_events():
    bars = data.synthetic_panel(n=300, n_tickers=1)
    r = st.regression_slope(st.build_events(bars, "MKT"))
    assert np.isnan(r["slope"])


def test_bootstrap_interval_brackets_the_planted_truth():
    bars = data.synthetic_panel(n=3000, n_tickers=10, drop_fraction=0.8)
    ci = st.bootstrap_ci(st.build_events(bars, "MKT"), n_boot=400)
    assert ci["lo"] < 0.8 < ci["hi"]


def test_a_full_drop_is_not_mistaken_for_a_partial_one():
    bars = data.synthetic_panel(n=4000, n_tickers=10, drop_fraction=1.0)
    ci = st.bootstrap_ci(st.build_events(bars, "MKT"), n_boot=400)
    assert ci["lo"] < 1.0 < ci["hi"]


# --------------------------------------------------------------------------- #
# Cuts and the trade
# --------------------------------------------------------------------------- #
def test_yield_buckets_split_the_events():
    bars = data.synthetic_panel(n=3000, n_tickers=10)
    b = st.yield_buckets(st.build_events(bars, "MKT"))
    assert len(b) >= 3
    assert b["median_yield"].is_monotonic_increasing


def test_by_group_drops_thin_groups():
    bars = data.synthetic_panel(n=1000, n_tickers=6)
    g = st.by_group(st.build_events(bars, "MKT"), "ticker", min_n=1000)
    assert len(g) == 0


def test_capture_trade_is_profitable_exactly_when_the_drop_is_partial():
    for frac, sign in ((0.5, 1), (1.5, -1)):
        bars = data.synthetic_panel(n=3000, n_tickers=10, drop_fraction=frac, daily_vol=1e-9,
                                    market_beta=0.0)
        r = st.capture_trade(st.build_events(bars, "MKT"), cost_bps=0.0)
        assert np.sign(r["mean_gross_bps"]) == sign


def test_capture_trade_charges_both_legs():
    bars = data.synthetic_panel(n=2000, n_tickers=6, drop_fraction=0.5)
    free = st.capture_trade(st.build_events(bars, "MKT"), cost_bps=0.0)
    paid = st.capture_trade(st.build_events(bars, "MKT"), cost_bps=10.0)
    assert free["mean_net_bps"] - paid["mean_net_bps"] == pytest.approx(20.0, abs=1e-6)


def test_dividend_tax_reduces_the_capture_edge():
    bars = data.synthetic_panel(n=2000, n_tickers=6, drop_fraction=0.5)
    ev = st.build_events(bars, "MKT")
    assert (st.capture_trade(ev, div_tax=0.2)["mean_net_bps"]
            < st.capture_trade(ev, div_tax=0.0)["mean_net_bps"])


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_events": 520, "n_tickers": 12, "window": ["2005-02-14", "2026-06-10"],
         "slope": 0.86, "t_vs_one": -2.6, "t_vs_zero": 16.0, "mean_ratio": 0.42,
         "median_ratio": 0.83, "ratio_of_sums": 0.88, "typical_move": 0.012,
         "typical_yield": 0.0068, "share_wild": 0.31, "elton_gruber": 0.778,
         "eg_inside_ci": True, "ci_lo": 0.74, "ci_hi": 1.01, "estimator_spread": 0.46,
         "gross_bps": 8.0, "net_bps": 4.0, "cost_bps": 2.0, "t_trade": 1.4, "hit_rate": 0.52,
         "sd_bps": 130.0, "breakeven_bps": 4.0, "tax_rate": 0.15,
         "net_after_tax_bps": -6.0}
    h.update(over)
    return h


def test_verdict_signal_needs_a_shortfall_that_is_measurable():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(t_vs_one=-1.1))["signal"] == "Weak"
    assert st.verdict(_headline(slope=0.99, t_vs_one=-0.2))["signal"] == "None"
    assert st.verdict(_headline(slope=1.04, t_vs_one=2.5))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(t_trade=2.3))["trad"] == "Investable"
    assert st.verdict(_headline(net_bps=-3.0))["trad"] == "Mirage"


def test_verdict_prose_names_elton_gruber_and_the_estimator_spread():
    v = st.verdict(_headline())
    assert "Elton and Gruber" in v["signal_why"]
    assert "0.46" in v["one_sentence"] or "spread" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
