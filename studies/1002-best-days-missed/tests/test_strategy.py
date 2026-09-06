"""Strategy tests for Study 1002 — the best-days statistic, taken apart."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bestdays import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
def test_total_return_compounds():
    r = pd.Series([0.1, 0.1, 0.1])
    assert st.total_return(r) == pytest.approx(1.1 ** 3 - 1)


def test_annualised_matches_a_known_case():
    r = pd.Series([0.0] * 252)
    assert st.annualised(r) == pytest.approx(0.0, abs=1e-12)
    r2 = pd.Series([0.001] * 504)
    assert st.annualised(r2) == pytest.approx(1.001 ** 252 - 1, rel=1e-9)


def test_dropping_the_best_days_lowers_the_return():
    r = st.synthetic_returns(n=3000)
    assert st.total_return(st.drop_extremes(r, 10, "best")) < st.total_return(r)


def test_dropping_the_worst_days_raises_the_return():
    r = st.synthetic_returns(n=3000)
    assert st.total_return(st.drop_extremes(r, 10, "worst")) > st.total_return(r)


def test_drop_extremes_preserves_the_length():
    """Replacing rather than deleting — otherwise the annualised figure is inflated."""
    r = st.synthetic_returns(n=2000)
    assert len(st.drop_extremes(r, 50, "best")) == len(r)


def test_drop_extremes_removes_exactly_the_right_days():
    r = pd.Series([0.05, -0.01, 0.02, -0.09, 0.01],
                  index=pd.bdate_range("2020-01-01", periods=5))
    out = st.drop_extremes(r, 1, "best")
    assert out.iloc[0] == 0.0
    assert (out.iloc[1:] == r.iloc[1:]).all()
    out2 = st.drop_extremes(r, 1, "worst")
    assert out2.iloc[3] == 0.0


def test_dropping_zero_days_changes_nothing():
    r = st.synthetic_returns(n=1000)
    assert st.total_return(st.drop_extremes(r, 0, "best")) == pytest.approx(
        st.total_return(r))


def test_the_missed_days_table_is_monotone():
    r = st.synthetic_returns(n=6000)
    t = st.missed_days_table(r, counts=(0, 5, 10, 20, 50))
    assert t["miss_best_cagr"].is_monotonic_decreasing
    assert t["miss_worst_cagr"].is_monotonic_increasing


def test_cash_rate_softens_the_penalty():
    """An investor out of the market holds cash, not a void. The brochure forgets this."""
    r = st.synthetic_returns(n=6000)
    zero = st.missed_days_table(r, counts=(50,), cash_rate=0.0)
    cash = st.missed_days_table(r, counts=(50,), cash_rate=0.04)
    assert cash.loc[50, "miss_best_cagr"] > zero.loc[50, "miss_best_cagr"]


# --------------------------------------------------------------------------- #
# The omitted half
# --------------------------------------------------------------------------- #
def test_the_best_days_are_bigger_in_percent_but_smaller_in_logs():
    """The mechanism, and it is not the obvious one.

    On the S&P 500 the ten best days are LARGER in percentage terms than the ten worst, so
    "crashes are bigger than rallies" does not explain the asymmetry here. Compounding is
    multiplicative, so the scale that matters is log(1+x) — and on that scale the worst days
    are the larger ones. Both facts are asserted, because the study's explanation depends on
    the pair.
    """
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    a = st.asymmetry(r, 10)
    assert a["best_bigger_in_percent"]
    assert a["worst_bigger_than_best"]
    assert abs(a["log_worst"]) > a["log_best"]


def test_the_log_scale_is_what_compounding_actually_uses():
    """A direct demonstration, independent of any market data."""
    up, down = 0.0874, -0.0836
    assert up > abs(down)                                  # bigger in percent
    assert abs(np.log1p(down)) > np.log1p(up)              # smaller in logs
    r = pd.Series([up, down] + [0.0004] * 500)
    without_up = st.total_return(st.drop_extremes(r, 1, "best"))
    without_down = st.total_return(st.drop_extremes(r, 1, "worst"))
    base = st.total_return(r)
    assert (without_down - base) > (base - without_up)


def test_missing_the_worst_days_helps_more_than_missing_the_best_hurts():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    a = st.asymmetry(r, 10)
    assert a["benefit_of_missing_worst"] > a["cost_of_missing_best"]
    assert a["ratio"] > 1.0


def test_the_asymmetry_holds_across_markets():
    px = data.load_prices()
    ratios = []
    for tk in (data.EQUITY, data.NASDAQ, data.SMALL, data.EAFE, data.EM):
        s = px[tk].dropna().pct_change().dropna()
        if len(s) > 1500:
            ratios.append(st.asymmetry(s, 10)["ratio"])
    assert len(ratios) >= 4
    assert np.median(ratios) > 1.0


def test_asymmetry_reports_the_base_case_intact():
    r = st.synthetic_returns(n=4000)
    a = st.asymmetry(r, 10)
    assert a["base_cagr"] == pytest.approx(st.annualised(r))


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def test_extreme_days_returns_both_kinds():
    r = st.synthetic_returns(n=3000)
    ex = st.extreme_days(r, 10)
    assert len(ex) == 20
    assert (ex["kind"] == "best").sum() == 10
    assert ex["date"].is_monotonic_increasing


def test_the_best_and_worst_days_are_neighbours_on_the_real_tape():
    """The finding that makes the brochure's counterfactual impossible."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    c = st.clustering_stats(r, 10, n_shuffles=300)
    assert c["median_gap"] < c["shuffled_median_gap"]
    assert c["p_value"] < 0.05


def test_shuffling_destroys_the_clustering_but_keeps_the_statistic():
    """The control: the fat tail survives the shuffle, the clustering does not."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    rng = np.random.default_rng(0)
    sh = pd.Series(rng.permutation(r.to_numpy()), index=r.index)
    assert st.asymmetry(sh, 10)["cost_of_missing_best"] == pytest.approx(
        st.asymmetry(r, 10)["cost_of_missing_best"], rel=1e-9)
    assert st.clustering_stats(sh, 10, 200)["median_gap"] > \
        st.clustering_stats(r, 10, 200)["median_gap"]


def test_clustering_needs_volatility_clustering_to_appear():
    """The mechanism, isolated: no vol clustering, no proximity."""
    a = st.clustering_stats(st.synthetic_returns(n=8000, clustered=True), 10, 200)
    b = st.clustering_stats(st.synthetic_returns(n=8000, clustered=False), 10, 200)
    assert a["ratio"] < b["ratio"]


def test_iid_returns_show_no_clustering_against_their_own_shuffle():
    c = st.clustering_stats(st.synthetic_returns(n=8000, clustered=False), 10, 300)
    assert c["p_value"] > 0.01


def test_extreme_days_arrive_in_high_volatility():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    v = st.volatility_context(r, 10)
    assert v["best_vol_ratio"] > 1.5
    assert v["worst_vol_ratio"] > 1.5


def test_the_best_days_happen_inside_drawdowns():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    d = st.drawdown_context(r, 10)
    assert d["best_median_drawdown"] < d["typical_drawdown"]
    assert d["best_median_drawdown"] < -0.05


# --------------------------------------------------------------------------- #
# What a real timer would need
# --------------------------------------------------------------------------- #
def test_missing_random_days_costs_almost_nothing():
    """The correct null, and the whole point."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    t = st.out_of_market_cost(r, fractions=(0.001,), n_draws=200)
    row = t.iloc[0]
    assert row["random_cost"] < row["worst_case_cost"] / 4


def test_the_random_cost_grows_with_the_fraction_missed():
    r = st.synthetic_returns(n=6000)
    t = st.out_of_market_cost(r, fractions=(0.01, 0.05, 0.25), n_draws=100)
    assert t["random_cost"].is_monotonic_increasing


def test_the_timing_frontier_rises_with_accuracy():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    f = st.timing_frontier(r, 0.20, (0.40, 0.50, 0.70, 1.0), n_draws=100)
    assert f["median_cagr"].is_monotonic_increasing


def test_a_perfect_timer_beats_buy_and_hold():
    """A sanity check on the machinery: perfect accuracy must win."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    f = st.timing_frontier(r, 0.20, (1.0,), n_draws=60)
    assert f.loc[1.0, "median_cagr"] > f.loc[1.0, "buy_and_hold"]


def test_a_random_timer_loses_the_premium_it_sits_out():
    """No skill means no reason to be out — and the cost is the premium forgone."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    rate = st.down_day_share(r)
    f = st.timing_frontier(r, 0.20, (rate,), n_draws=150)
    assert f.iloc[0]["median_cagr"] < f.iloc[0]["buy_and_hold"]


def test_the_coin_flip_benchmark_is_the_down_day_share_not_one_half():
    """The definitional point the first version of this study got wrong."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    rate = st.down_day_share(r)
    assert 0.40 < rate < 0.50
    assert abs(rate - 0.5) > 0.02


def test_the_breakeven_sits_just_above_random_selection():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    be = st.breakeven_hit_rate(r, 0.20)
    rate = st.down_day_share(r)
    assert be > rate
    assert be - rate < 0.15


def test_the_breakeven_is_stable_across_how_much_the_timer_trades():
    """A good sign: the required EDGE is a property of the market, not of the trading rate."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    bes = [st.breakeven_hit_rate(r, f) for f in (0.05, 0.20, 0.35)]
    assert max(bes) - min(bes) < 0.05


def test_being_below_random_is_costlier_than_being_equally_above_is_profitable():
    """The steepness that makes a small required edge unforgiving."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna()
    be = st.breakeven_hit_rate(r, 0.20)
    f = st.timing_frontier(r, 0.20, (be - 0.05, be, be + 0.05), n_draws=150)
    base = float(f["buy_and_hold"].iloc[0])
    loss = base - float(f.iloc[0]["median_cagr"])
    gain = float(f.iloc[2]["median_cagr"]) - base
    assert loss > 0 and gain > 0


# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #
def test_both_synthetic_worlds_have_similar_unconditional_tails():
    """The comparison is only valid if the fat tail is held constant."""
    a = st.synthetic_returns(n=20000, clustered=True)
    b = st.synthetic_returns(n=20000, clustered=False)
    assert abs(a.kurtosis() - b.kurtosis()) < a.kurtosis() * 1.2
    assert abs(a.std() / b.std() - 1) < 0.35


def test_only_the_clustered_world_has_persistent_volatility():
    a = st.synthetic_returns(n=8000, clustered=True).abs()
    b = st.synthetic_returns(n=8000, clustered=False).abs()
    assert a.autocorr(1) > 0.1
    assert abs(b.autocorr(1)) < 0.05


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "years": 33.4, "n_days": 8412, "base_cagr": 0.0991,
         "cost_of_missing_best": 0.0221, "benefit_of_missing_worst": 0.0288,
         "asym_ratio": 1.30, "mean_best": 0.0748, "mean_worst": -0.0891,
         "log_best": 0.0838, "log_worst": -0.0873,
         "median_gap": 3.0, "shuffled_gap": 214.0, "cluster_p": 0.002,
         "best_drawdown": -0.244, "best_vol_ratio": 2.9,
         "out_fraction": 0.20, "days_out": 1682, "coin_flip_rate": 0.4516,
         "breakeven_hit_rate": 0.4768, "timing_edge_needed": 0.0252,
         "coin_flip_cagr": 0.0887, "below_gap": 5.0, "below_cagr": 0.0434,
         "random_days": 84, "random_fraction": 0.01, "random_cost": 0.0009,
         "worst_case_cost": 0.0721}
    h.update(over)
    return h


def test_verdict_signal_needs_both_the_arithmetic_and_the_asymmetry():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(asym_ratio=0.8))["signal"] == "Mixed"
    assert st.verdict(_headline(cost_of_missing_best=0.001))["signal"] == "Busted"


def test_verdict_tradability_keys_off_the_edge_over_random_not_over_one_half():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(timing_edge_needed=0.005))["trad"] == "Investable"
    assert st.verdict(_headline(timing_edge_needed=0.12))["trad"] == "Mirage"


def test_verdict_prose_corrects_the_obvious_wrong_mechanism():
    v = st.verdict(_headline())
    assert "The brochure stops there" in v["signal_why"]
    assert "log(1+x)" in v["signal_why"]
    assert "simply false here" in v["signal_why"]
    assert "not 50%" in v["trad_why"]
    assert "at random" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
