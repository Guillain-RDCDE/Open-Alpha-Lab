"""Strategy tests for Study 995 — currency arithmetic checked exactly."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from whosesharpe import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The conversion, by hand
# --------------------------------------------------------------------------- #
def test_conversion_is_the_exact_ratio_not_the_difference():
    idx = pd.bdate_range("2020-01-01", periods=3)
    a = pd.Series([0.10, 0.10, 0.10], index=idx)
    f = pd.Series([0.05, 0.05, 0.05], index=idx)
    c = st.convert_return(a, f)
    assert c.iloc[0] == pytest.approx(1.10 / 1.05 - 1)
    assert c.iloc[0] != pytest.approx(0.05)          # NOT the naive difference


def test_a_flat_currency_leaves_the_return_alone():
    idx = pd.bdate_range("2020-01-01", periods=100)
    a = pd.Series(np.linspace(-0.02, 0.02, 100), index=idx)
    f = pd.Series(0.0, index=idx)
    assert np.allclose(st.convert_return(a, f).to_numpy(), a.to_numpy())


def test_a_rising_home_currency_reduces_the_foreign_return():
    idx = pd.bdate_range("2020-01-01", periods=50)
    a = pd.Series(0.01, index=idx)
    strong = pd.Series(0.005, index=idx)
    assert st.convert_return(a, strong).mean() < a.mean()


def test_the_approximation_error_compounds_over_a_decade():
    """Why the ratio form matters: r - f is fine for a day and wrong for ten years."""
    n = 2520
    idx = pd.bdate_range("2010-01-01", periods=n)
    a = pd.Series(0.08 / 252, index=idx)
    errs = []
    for annual_fx in (0.02, 0.10, 0.30, 0.60):
        f = pd.Series(annual_fx / 252, index=idx)
        exact = float((1 + st.convert_return(a, f)).prod())
        naive = float((1 + (a - f)).prod())
        errs.append(abs(exact - naive) / exact)
    # small for a placid currency, and growing steadily with its drift
    assert errs == sorted(errs)
    assert errs[-1] > 20 * errs[0]


def test_conversion_intersects_the_calendars():
    idx = pd.bdate_range("2020-01-01", periods=100)
    a = pd.Series(0.001, index=idx)
    f = pd.Series(0.001, index=idx[20:])
    assert len(st.convert_return(a, f)) == 80


# --------------------------------------------------------------------------- #
# Sharpe and stats
# --------------------------------------------------------------------------- #
def test_sharpe_matches_its_definition():
    rng = np.random.default_rng(995)
    x = pd.Series(rng.normal(0.0004, 0.01, 5000))
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert st.sharpe(x) == pytest.approx(expected)


def test_sharpe_is_nan_on_a_constant_series():
    assert np.isnan(st.sharpe(pd.Series([0.001] * 500)))


def test_stats_block_subtracts_the_risk_free_rate():
    idx = pd.bdate_range("2010-01-01", periods=2520)
    r = pd.Series(0.0004, index=idx)
    zero = st.stats_block(r, pd.Series(0.0, index=idx))
    paid = st.stats_block(r, pd.Series(0.0002, index=idx))
    assert paid["excess_ann"] < zero["excess_ann"]
    assert paid["cagr"] == pytest.approx(zero["cagr"])     # the return itself is unchanged


def test_stats_block_declines_on_a_short_series():
    assert "sharpe" not in st.stats_block(pd.Series([0.01] * 20))


# --------------------------------------------------------------------------- #
# The variance channel
# --------------------------------------------------------------------------- #
def test_uncorrelated_currency_adds_variance_as_the_identity_says():
    w = st.synthetic_world(n=20000, corr=0.0, asset_vol=0.16, fx_vol=0.10)
    d = st.variance_decomposition(w["asset"], w["fx"])
    assert d["predicted_var"] == pytest.approx(d["realised_var"], rel=0.05)
    assert d["vol_ratio"] == pytest.approx(np.sqrt(0.16 ** 2 + 0.10 ** 2) / 0.16, rel=0.05)


def test_a_positively_correlated_currency_can_reduce_volatility():
    """The one case where currency exposure is a hedge rather than a cost."""
    low = st.variance_decomposition(*[st.synthetic_world(n=20000, corr=0.9,
                                                         asset_vol=0.16, fx_vol=0.10)[k]
                                      for k in ("asset", "fx")])
    assert low["vol_ratio"] < 1.0


def test_volatility_always_rises_when_the_currency_is_uncorrelated():
    for fx_vol in (0.05, 0.10, 0.20):
        w = st.synthetic_world(n=10000, corr=0.0, fx_vol=fx_vol)
        assert st.variance_decomposition(w["asset"], w["fx"])["vol_ratio"] > 1.0


def test_the_variance_identity_reports_its_own_approximation_error():
    w = st.synthetic_world(n=10000, corr=0.3)
    d = st.variance_decomposition(w["asset"], w["fx"])
    assert abs(d["approximation_error"]) < d["realised_var"] * 0.1


def test_variance_decomposition_declines_on_a_short_series():
    w = st.synthetic_world(n=100)
    assert "vol_ratio" not in st.variance_decomposition(w["asset"], w["fx"])


# --------------------------------------------------------------------------- #
# The optimal hedge ratio
# --------------------------------------------------------------------------- #
def test_the_optimal_hedge_is_one_when_asset_and_currency_are_independent():
    w = st.synthetic_world(n=20000, corr=0.0)
    assert st.hedge_ratio_that_minimises_variance(w["asset"], w["fx"]) == pytest.approx(
        1.0, abs=0.1)


def test_the_optimal_hedge_falls_below_one_when_they_co_move():
    """Campbell et al. (2010): full hedging overshoots when the asset moves with the currency."""
    w = st.synthetic_world(n=20000, corr=-0.5, asset_vol=0.16, fx_vol=0.10)
    assert st.hedge_ratio_that_minimises_variance(w["asset"], w["fx"]) < 0.9


def test_the_optimal_hedge_exceeds_one_in_the_other_direction():
    w = st.synthetic_world(n=20000, corr=0.5, asset_vol=0.16, fx_vol=0.10)
    assert st.hedge_ratio_that_minimises_variance(w["asset"], w["fx"]) > 1.1


def test_optimal_hedge_is_nan_on_a_short_series():
    w = st.synthetic_world(n=100)
    assert np.isnan(st.hedge_ratio_that_minimises_variance(w["asset"], w["fx"]))


# --------------------------------------------------------------------------- #
# The Sharpe gap and its decomposition
# --------------------------------------------------------------------------- #
def test_an_identical_currency_leaves_the_sharpe_alone():
    w = st.synthetic_world(n=10000, fx_vol=1e-9, fx_drift=0.0, rate_gap=0.0)
    d = st.decompose_sharpe_gap(w["asset"], w["fx"], w["usd_rf"], w["local_rf"])
    assert abs(d["gap"]) < 0.05


def _paired(n=40000, fx_vol=0.12, fx_drift=0.0, seed=995):
    """One fixed asset path, plus an independent currency drawn on top of it.

    Pairing matters here. ``synthetic_world`` draws the asset and the currency jointly, so
    changing ``fx_vol`` changes the asset path too and the comparison picks up that difference
    as well as the one being measured. Holding the asset fixed isolates the currency's effect
    exactly, which is the difference between a test that can see a 0.02 Sharpe change and one
    that cannot.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-03", periods=n)
    a = pd.Series(rng.normal(0.08 / 252, 0.16 / np.sqrt(252), n), index=idx)
    # Build the currency in LOG space and pin its realised log drift. Two reasons, both
    # necessary to isolate the variance channel.
    #
    # First, the realised drift must be fixed rather than merely zero in expectation: over any
    # finite sample a currency wanders, that wandering is the drift channel, and it is an order
    # of magnitude larger than the variance channel and pure luck.
    #
    # Second, it must be the LOG drift. A currency that is arithmetically driftless has a log
    # drift of about -sigma^2/2, and converting through it hands the foreign investor that
    # +sigma^2/2 back as genuine compounded return. At a 35% currency volatility that term is
    # 6% a year and it swamps the variance effect — so "arithmetically driftless" is not the
    # neutral currency it sounds like.
    lf = pd.Series(rng.normal(0.0, fx_vol / np.sqrt(252), n), index=idx)
    lf = lf - lf.mean() + np.log1p(fx_drift / 252)
    f = np.expm1(lf)
    rf = pd.Series(0.02 / 252, index=idx)
    return a, f, rf


def test_an_uncorrelated_currency_lowers_the_foreign_sharpe():
    """Pure variance channel: more risk, same compounded return, lower Sharpe."""
    gaps = []
    for k in range(4):
        a, f, rf = _paired(fx_vol=0.12, seed=995 + k)
        gaps.append(st.geometric_sharpe(st.convert_return(a, f), rf)
                    - st.geometric_sharpe(a, rf))
    assert np.mean(gaps) < 0
    assert sum(g < 0 for g in gaps) >= 3


def test_the_arithmetic_sharpe_is_fooled_by_the_currency_variance():
    """The trap that makes the log version necessary rather than fastidious.

    The arithmetic mean of (1+a)/(1+f) - 1 carries a convexity term of about var(f). So a more
    volatile currency inflates the numerator at the same time as the denominator, and an
    arithmetic Sharpe can stay flat — or rise — while the investor is unambiguously worse off.
    The log Sharpe has no such term and falls, as it should.
    """
    arith, geo = [], []
    for k in range(4):
        a, f, rf = _paired(fx_vol=0.35, seed=995 + k)
        conv = st.convert_return(a, f)
        arith.append(st.sharpe(conv - rf.reindex(conv.index))
                     - st.sharpe(a - rf.reindex(a.index)))
        geo.append(st.geometric_sharpe(conv, rf) - st.geometric_sharpe(a, rf))
    assert np.mean(geo) < 0                       # the honest measure falls
    assert np.mean(arith) > np.mean(geo)          # the arithmetic one is flattered
    # The size of the flattery, which is what a reader should carry away: at a 35% currency
    # volatility the arithmetic Sharpe understates the damage by roughly a third of the
    # damage itself.
    assert np.mean(arith) - np.mean(geo) > 0.01


def test_a_falling_home_currency_raises_the_foreign_sharpe():
    w = st.synthetic_world(n=20000, corr=0.0, fx_vol=0.06, fx_drift=-0.04)
    up = st.decompose_sharpe_gap(w["asset"], w["fx"], w["usd_rf"], w["local_rf"])
    w2 = st.synthetic_world(n=20000, corr=0.0, fx_vol=0.06, fx_drift=0.0)
    flat = st.decompose_sharpe_gap(w2["asset"], w2["fx"], w2["usd_rf"], w2["local_rf"])
    assert up["gap"] > flat["gap"]
    assert up["drift_term"] > flat["drift_term"]


def test_a_higher_home_cash_rate_lowers_the_foreign_sharpe():
    """The channel everyone forgets: your own risk-free rate is the bar you clear."""
    low = st.synthetic_world(n=20000, fx_vol=0.08, rate_gap=0.0)
    high = st.synthetic_world(n=20000, fx_vol=0.08, rate_gap=0.04)
    a = st.decompose_sharpe_gap(low["asset"], low["fx"], low["usd_rf"], low["local_rf"])
    b = st.decompose_sharpe_gap(high["asset"], high["fx"], high["usd_rf"], high["local_rf"])
    assert b["sharpe_foreign"] < a["sharpe_foreign"]
    assert b["rate_term"] < a["rate_term"]


def test_the_decomposition_adds_up():
    w = st.synthetic_world(n=10000, corr=0.2, fx_drift=-0.02, rate_gap=0.01)
    d = st.decompose_sharpe_gap(w["asset"], w["fx"], w["usd_rf"], w["local_rf"])
    assert (d["drift_term"] + d["variance_term"] + d["rate_term"]
            == pytest.approx(d["gap"], abs=1e-9))


def test_decompose_declines_on_a_short_series():
    w = st.synthetic_world(n=250)
    assert "gap" not in st.decompose_sharpe_gap(w["asset"], w["fx"], w["usd_rf"],
                                                w["local_rf"])


# --------------------------------------------------------------------------- #
# Hedging
# --------------------------------------------------------------------------- #
def test_hedging_removes_most_of_the_currency_volatility():
    w = st.synthetic_world(n=10000, corr=0.0, fx_vol=0.12)
    conv = st.convert_return(w["asset"], w["fx"])
    hedged = st.hedged_return(w["asset"], w["fx"], w["usd_rf"], w["local_rf"], 1.0, 0.0)
    assert hedged.std() < conv.std() * 0.85
    # and it should land close to the unhedged ASSET's own volatility
    assert hedged.std() == pytest.approx(float(w["asset"].std()), rel=0.15)


def test_a_zero_hedge_ratio_changes_nothing():
    w = st.synthetic_world(n=5000)
    conv = st.convert_return(w["asset"], w["fx"])
    zero = st.hedged_return(w["asset"], w["fx"], w["usd_rf"], w["local_rf"], 0.0, 0.0)
    assert np.allclose(zero.to_numpy(), conv.reindex(zero.index).to_numpy())


def test_the_hedge_pays_the_interest_rate_differential():
    """A rolling forward hedge is not free, and its price is the rate gap."""
    cheap = st.synthetic_world(n=10000, fx_vol=0.10, rate_gap=0.0)
    dear = st.synthetic_world(n=10000, fx_vol=0.10, rate_gap=-0.04)   # home rates far lower
    a = st.hedged_return(cheap["asset"], cheap["fx"], cheap["usd_rf"], cheap["local_rf"],
                         1.0, 0.0).mean()
    b = st.hedged_return(dear["asset"], dear["fx"], dear["usd_rf"], dear["local_rf"],
                         1.0, 0.0).mean()
    assert b < a


def test_hedging_costs_reduce_the_hedged_return():
    w = st.synthetic_world(n=5000)
    free = st.hedged_return(w["asset"], w["fx"], w["usd_rf"], w["local_rf"], 1.0, 0.0)
    paid = st.hedged_return(w["asset"], w["fx"], w["usd_rf"], w["local_rf"], 1.0, 50.0)
    assert paid.mean() < free.mean()


def test_hedge_analysis_reports_every_ratio_and_the_optimum():
    w = st.synthetic_world(n=10000, corr=-0.3, fx_vol=0.10)
    ha = st.hedge_analysis(w["asset"], w["fx"], w["usd_rf"], w["local_rf"])
    assert set([0.0, 0.5, 1.0]).issubset(set(ha["table"].index))
    assert np.isfinite(ha["optimal_ratio"])
    assert ha["table"].loc[1.0, "vol"] < ha["table"].loc[0.0, "vol"]


# --------------------------------------------------------------------------- #
# The panel and rankings
# --------------------------------------------------------------------------- #
def test_sharpe_by_currency_includes_the_home_row():
    w = st.synthetic_world(n=6000)
    tbl = st.sharpe_by_currency(w["asset"], {"EUR": w["fx"]}, w["usd_rf"],
                                {"EUR": w["local_rf"]})
    assert list(tbl.index) == ["USD", "EUR"]
    assert np.isfinite(tbl.loc["USD", "sharpe"])


def test_currencies_spread_the_sharpe_of_one_asset():
    base = st.synthetic_world(n=8000, corr=0.0, fx_vol=0.12, fx_drift=-0.03)
    other = st.synthetic_world(n=8000, corr=0.0, fx_vol=0.12, fx_drift=+0.03, seed=1)
    tbl = st.sharpe_by_currency(base["asset"], {"A": base["fx"], "B": other["fx"]},
                                base["usd_rf"],
                                {"A": base["local_rf"], "B": other["local_rf"]})
    assert tbl["sharpe"].max() - tbl["sharpe"].min() > 0.1


def test_ranking_stability_returns_one_column_per_currency():
    w = st.synthetic_world(n=6000)
    w2 = st.synthetic_world(n=6000, asset_drift=0.03, seed=2)
    r = st.ranking_stability({"X": w["asset"], "Y": w2["asset"]},
                             {"EUR": w["fx"]}, w["usd_rf"])
    assert list(r.columns) == ["USD", "EUR"]
    assert "_rank_spread" in r.index


def test_implied_foreign_rate_moves_with_the_currency_drift():
    w = st.synthetic_world(n=6000, fx_drift=0.05)
    w2 = st.synthetic_world(n=6000, fx_drift=-0.05, seed=3)
    a = st.implied_foreign_rate(w["fx"], w["usd_rf"]).mean()
    b = st.implied_foreign_rate(w2["fx"], w2["usd_rf"]).mean()
    assert a < b        # a strengthening currency implies a LOWER local rate


def test_implied_foreign_rate_falls_back_on_a_short_series():
    w = st.synthetic_world(n=100)
    assert (st.implied_foreign_rate(w["fx"], w["usd_rf"]) == 0).all()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "years": 18.0, "n_assets": 5, "sharpe_usd": 0.62,
         "sharpe_min": 0.44, "sharpe_max": 0.79, "sharpe_spread": 0.35,
         "worst_currency": "CHF", "best_currency": "JPY", "vol_usd": 0.181,
         "vol_median_foreign": 0.203, "median_corr": 0.06, "rank_spread": 2.0,
         "hedge_helps_share": 0.67, "median_hedge_gain": 0.08,
         "median_optimal_ratio": 0.88}
    h.update(over)
    return h


def test_verdict_signal_needs_a_spread_and_a_reordering():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(rank_spread=1.0))["signal"] == "Weak"
    assert st.verdict(_headline(sharpe_spread=0.05))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(hedge_helps_share=0.3))["trad"] == "Partial"
    assert st.verdict(_headline(hedge_helps_share=0.0))["trad"] == "Mirage"


def test_verdict_prose_names_all_three_channels():
    v = st.verdict(_headline())
    for word in ("variance", "drift", "rate"):
        assert word in v["signal_why"]
    assert "interest-rate differential" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}


def test_verdict_switches_on_whether_a_full_hedge_overshoots():
    over = st.verdict(_headline(median_optimal_ratio=0.85))["trad_why"]
    fine = st.verdict(_headline(median_optimal_ratio=1.02))["trad_why"]
    assert "overshoots" in over
    assert "close to right" in fine
