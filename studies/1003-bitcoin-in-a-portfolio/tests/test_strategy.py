"""Strategy tests for Study 1003 — the bitcoin sleeve, and whether it is estimable."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from onepercent import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_sixty_forty_is_a_weighted_average():
    idx = pd.bdate_range("2020-01-01", periods=50)
    df = pd.DataFrame({"SPY": np.full(50, 0.01), "AGG": np.full(50, 0.002)}, index=idx)
    s = st.sixty_forty(df, "SPY", "AGG")
    assert s.iloc[0] == pytest.approx(0.6 * 0.01 + 0.4 * 0.002)


def test_a_zero_sleeve_is_the_base_portfolio():
    w = st.synthetic_pair(n=800)
    out = st.sleeve(w["base"], w["asset"], 0.0)
    assert np.allclose(out.to_numpy(), w["base"].to_numpy())


def test_a_full_sleeve_is_the_asset():
    w = st.synthetic_pair(n=800)
    out = st.sleeve(w["base"], w["asset"], 1.0)
    assert np.allclose(out.to_numpy(), w["asset"].to_numpy())


def test_the_sleeve_is_funded_from_the_base_not_added_on_top():
    """A 5% sleeve means 95% of the old portfolio, not 105% of the money."""
    w = st.synthetic_pair(n=800)
    out = st.sleeve(w["base"], w["asset"], 0.05)
    expected = 0.95 * w["base"] + 0.05 * w["asset"]
    assert np.allclose(out.to_numpy(), expected.to_numpy())


def test_stats_annualise_correctly():
    r = pd.Series([0.001] * 504, index=pd.bdate_range("2020-01-01", periods=504))
    s = st.stats(r)
    assert s["cagr"] == pytest.approx(1.001 ** 252 - 1, rel=1e-9)
    assert s["vol"] == pytest.approx(0.0, abs=1e-12)


def test_stats_declines_on_a_short_series():
    assert st.stats(pd.Series([0.01] * 5)) == {}


def test_the_weight_sweep_covers_every_weight():
    w = st.synthetic_pair(n=1500)
    sw = st.weight_sweep(w["base"], w["asset"], (0.0, 0.05, 0.20))
    assert list(sw.index) == [0.0, 0.05, 0.20]
    assert sw["vol"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# The calendar, which is not a detail
# --------------------------------------------------------------------------- #
def test_the_raw_panel_really_does_mix_two_calendars():
    """The problem exists — asserted, so the fix below is not solving an imaginary one."""
    px = data.load_prices()
    btc = px[data.BTC].dropna()
    per_year = len(btc) / ((btc.index[-1] - btc.index[0]).days / 365.25)
    assert per_year > 300                      # bitcoin trades weekends
    # over the OVERLAPPING window only — SPY's tape starts two decades earlier
    spy = px[data.EQUITY].dropna()
    spy = spy[spy.index >= btc.index[0]]
    assert len(btc) > 1.25 * len(spy)


def test_alignment_puts_every_series_on_the_equity_calendar():
    px = st.align_to_equity_calendar(data.load_prices(), data.EQUITY)
    btc = px[data.BTC].dropna()
    per_year = len(btc) / ((btc.index[-1] - btc.index[0]).days / 365.25)
    assert 240 < per_year < 265                # trading days, not calendar days
    assert px.index.equals(data.load_prices()[data.EQUITY].dropna().index)


def test_alignment_folds_weekend_moves_into_the_next_session_not_away():
    """Forward-filling must preserve the cumulative move, not discard the weekend."""
    px = data.load_prices()
    aligned = st.align_to_equity_calendar(px, data.EQUITY)
    raw = px[data.BTC].dropna()
    a = aligned[data.BTC].dropna()
    assert a.iloc[-1] == pytest.approx(raw.reindex(a.index).ffill().iloc[-1])
    total_raw = raw.iloc[-1] / raw.reindex(a.index).dropna().iloc[0]
    total_aligned = a.iloc[-1] / a.iloc[0]
    assert total_aligned == pytest.approx(total_raw, rel=1e-6)


def test_the_calendar_choice_changes_the_annualised_volatility():
    """Documented because getting it wrong flatters bitcoin, and it is a common error."""
    px = data.load_prices()
    r = px[data.BTC].dropna().pct_change().dropna()
    vol_252 = r.std(ddof=1) * np.sqrt(252)
    vol_365 = r.std(ddof=1) * np.sqrt(365)
    assert vol_365 > vol_252 * 1.15


def test_alignment_leaves_a_series_that_never_trades_weekends_untouched():
    px = data.load_prices()
    aligned = st.align_to_equity_calendar(px, data.EQUITY)
    a = px[data.EQUITY].dropna()
    b = aligned[data.EQUITY].dropna()
    assert a.equals(b.reindex(a.index))


# --------------------------------------------------------------------------- #
# The optimiser
# --------------------------------------------------------------------------- #
def test_the_optimiser_finds_a_planted_weight_on_a_long_sample():
    """It is unbiased. That is not the same as useful, which the next tests show."""
    found = []
    for k in range(4):
        w = st.synthetic_pair(n=25000, true_weight=0.05, seed=1003 + k)
        found.append(st.optimal_weight(w["base"], w["asset"], 0.30))
    assert abs(np.mean(found) - 0.05) < 0.02


def test_the_optimiser_is_hopeless_on_a_realistic_sample():
    """The headline of the study, on planted data where the truth is known exactly."""
    found = []
    for k in range(12):
        w = st.synthetic_pair(n=2500, true_weight=0.05, seed=1003 + k)
        found.append(st.optimal_weight(w["base"], w["asset"], 0.30))
    assert np.std(found) > 0.03          # a spread comparable to the answer itself


def test_the_optimiser_returns_zero_for_a_worthless_asset():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=4000)
    base = pd.Series(rng.normal(0.0004, 0.008, 4000), index=idx)
    junk = pd.Series(rng.normal(-0.002, 0.05, 4000), index=idx)
    assert st.optimal_weight(base, junk, 0.30) < 0.02


def test_the_optimiser_respects_its_cap():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    base = pd.Series(rng.normal(0.0001, 0.008, 3000), index=idx)
    great = pd.Series(rng.normal(0.004, 0.010, 3000), index=idx)
    assert st.optimal_weight(base, great, 0.20) == pytest.approx(0.20, abs=1e-9)


def test_the_objective_curve_is_a_plateau_not_a_peak():
    """Why the argmax is not worth arguing about."""
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    f = st.flatness(st.objective_curve(base, btc, 0.30), tol=0.01)
    assert f["plateau_width"] > 0.03


def test_flatness_handles_a_degenerate_curve():
    assert st.flatness(pd.DataFrame({"value": [np.nan, np.nan]}, index=[0.0, 0.1])) == {}


# --------------------------------------------------------------------------- #
# Estimability
# --------------------------------------------------------------------------- #
def test_the_bootstrap_interval_is_very_wide():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    b = st.weight_standard_error(base, btc, 0.50, n_boot=120)
    assert b["p95"] - b["p05"] > 0.05


def test_the_bootstrap_declines_on_a_short_sample():
    w = st.synthetic_pair(n=100)
    assert st.weight_standard_error(w["base"], w["asset"]) == {}


def test_sample_needed_scales_with_the_square_of_volatility():
    """Doubling volatility quadruples the data required. The whole problem, in one line."""
    a = st.sample_needed(0.5, 0.35, 0.02)
    b = st.sample_needed(0.5, 0.70, 0.02)
    assert b["years_needed"] == pytest.approx(4 * a["years_needed"], rel=1e-9)


def test_bitcoin_needs_an_implausible_amount_of_history():
    px = data.load_prices()
    r = px[data.BTC].dropna().pct_change().dropna()
    vol = float(r.std(ddof=1) * np.sqrt(252))
    s = st.sample_needed(0.0, vol, 0.02)
    assert s["years_needed"] > 200


def test_conditioned_on_its_history_the_sample_prefers_five_to_one():
    """Not the result this study expected, and the reason it changed shape.

    Given bitcoin's realised return, a bootstrap does rank 5% above 1% decisively. The
    published small allocations therefore cannot be defended as "what the noisy data
    supports" — the data supports a much larger sleeve. What they actually rest on is a
    different expected return, which is what `implied_mean_for_weight` extracts.
    """
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    p = st.power_to_distinguish(base, btc, 0.01, 0.05, n_boot=200)
    assert p["share_b_wins"] > 0.8
    assert p["distinguishable"]


def test_power_to_distinguish_finds_no_difference_where_the_curve_is_flat():
    """The machinery can also report 'no' — checked, so the test above means something.

    A 0% sleeve against a 90% one straddles the optimum, so the two sit at similar heights on
    opposite sides of the curve and no ranking survives resampling.
    """
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    p = st.power_to_distinguish(base, btc, 0.0, 0.90, n_boot=200)
    assert not p["distinguishable"]


# --------------------------------------------------------------------------- #
# The inversion — what each recommendation assumes
# --------------------------------------------------------------------------- #
def test_the_optimiser_wants_far_more_than_anyone_recommends():
    """The finding the study is built on."""
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    assert st.optimal_weight(base, btc, 0.50) > 0.10


def test_the_optimal_weight_rises_with_the_assumed_mean():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    c = st.weight_vs_assumed_mean(base, btc, np.linspace(-0.2, 0.6, 17))
    assert c["optimal_weight"].is_monotonic_increasing


def test_recentring_preserves_volatility_and_changes_only_the_drift():
    """The mechanism must be a pure drift shift, or the inversion measures the wrong thing."""
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    btc = rets[data.BTC].dropna().to_numpy(dtype=float)
    daily = np.log1p(btc).mean()
    shifted = np.expm1(np.log1p(btc) - daily + np.log1p(0.05) / 252)
    assert np.std(np.log1p(shifted), ddof=1) == pytest.approx(
        np.std(np.log1p(btc), ddof=1), rel=1e-9)
    got = np.expm1(np.log1p(shifted).sum() * 252 / len(shifted))
    assert got == pytest.approx(0.05, rel=1e-6)


def test_a_two_percent_sleeve_implies_roughly_no_expected_return():
    """The headline inversion: the industry number is the zero-expected-return answer."""
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    imp = st.implied_mean_for_weight(base, btc, (0.01, 0.02, 0.05))
    assert abs(imp.loc[0.02, "implied_mean"]) < 0.06
    assert imp.loc[0.01, "implied_mean"] < imp.loc[0.02, "implied_mean"]
    assert imp.loc[0.05, "implied_mean"] > imp.loc[0.02, "implied_mean"]


def test_the_implied_means_are_far_below_the_realised_one():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    realised = np.expm1(np.log1p(btc).sum() * 252 / len(btc))
    imp = st.implied_mean_for_weight(base, btc, (0.01, 0.02, 0.05))
    assert (imp["implied_mean"] < realised - 0.20).all()


def test_implied_mean_declines_on_a_short_sample():
    w = st.synthetic_pair(n=100)
    assert st.implied_mean_for_weight(w["base"], w["asset"]).empty
    assert st.weight_vs_assumed_mean(w["base"], w["asset"]).empty


def test_mean_uncertainty_is_wide_enough_to_matter():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    m = st.weight_with_mean_uncertainty(base, btc, 0.50, n_draws=120)
    assert m["mean_se"] > 0.10                       # a standard error of 10+ points
    assert m["p95"] - m["p05"] > 0.05


def test_mean_uncertainty_declines_on_a_short_sample():
    w = st.synthetic_pair(n=100)
    assert st.weight_with_mean_uncertainty(w["base"], w["asset"]) == {}


def test_power_to_distinguish_declines_on_a_short_sample():
    w = st.synthetic_pair(n=100)
    assert st.power_to_distinguish(w["base"], w["asset"]) == {}


# --------------------------------------------------------------------------- #
# Out of sample and implementation
# --------------------------------------------------------------------------- #
def test_the_walk_forward_allocator_changes_its_mind():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    wf = st.walk_forward_weights(base, btc, 3.0, 63, 0.20)
    assert len(wf) > 5
    assert wf["weight"].max() - wf["weight"].min() > 0.05


def test_the_walk_forward_allocator_pays_for_its_turnover():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    free = st.walk_forward_weights(base, btc, 3.0, 63, 0.20, cost_bps=0.0)
    paid = st.walk_forward_weights(base, btc, 3.0, 63, 0.20, cost_bps=100.0)
    assert paid["realised"].sum() < free["realised"].sum()


def test_walk_forward_declines_when_there_is_no_room():
    w = st.synthetic_pair(n=300)
    assert st.walk_forward_weights(w["base"], w["asset"], 3.0).empty
    assert st.walk_forward_series(w["base"], w["asset"], 3.0).empty


def test_never_rebalancing_lets_the_sleeve_take_over():
    """The '2% allocation' whose track record is quoted may never have been 2%."""
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    rb = st.rebalancing_matters(base, btc, 0.02, (21, 10_000))
    assert rb.loc[21, "max_weight_reached"] < 0.05          # monthly keeps it near 2%
    assert rb.loc[10_000, "max_weight_reached"] > 2 * rb.loc[21, "max_weight_reached"]


def test_rebalancing_frequency_changes_the_realised_risk():
    px = data.load_prices()
    rets = st.align_to_equity_calendar(px, data.EQUITY).pct_change()
    base = st.sixty_forty(rets, data.EQUITY, data.BONDS)
    btc = rets[data.BTC].dropna()
    rb = st.rebalancing_matters(base, btc, 0.02, (21, 10_000))
    assert rb.loc[10_000, "vol"] > rb.loc[21, "vol"]


# --------------------------------------------------------------------------- #
# The synthetic world
# --------------------------------------------------------------------------- #
def test_the_synthetic_pair_has_the_volatility_it_claims():
    w = st.synthetic_pair(n=40000, asset_vol=0.70)
    assert w["asset"].std(ddof=1) * np.sqrt(252) == pytest.approx(0.70, rel=0.05)


def test_the_planted_optimum_is_where_it_says_it_is():
    """Validates the mu-solving, without which the whole control is meaningless."""
    for true_w in (0.02, 0.05, 0.10):
        found = [st.optimal_weight(w["base"], w["asset"], 0.30) for w in
                 (st.synthetic_pair(n=30000, true_weight=true_w, seed=1003 + k)
                  for k in range(3))]
        assert abs(np.mean(found) - true_w) < 0.025


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"as_of": "2026-06-30", "years": 11.7, "base_cagr": 0.0712, "base_vol": 0.1041,
         "base_sharpe": 0.684, "base_dd": -0.211, "best_weight": 0.034,
         "best_sharpe": 0.812, "best_cagr": 0.0921, "best_dd": -0.238,
         "plateau_lo": 0.010, "plateau_hi": 0.072, "calendar_inflation": 0.20,
         "boot_p05": 0.06, "boot_p95": 0.50, "boot_at_zero": 0.00, "boot_at_cap": 0.06,
         "max_weight": 0.50, "share_5_beats_1": 0.99, "diff_5_1": 0.162,
         "diff_sd": 0.084, "btc_vol": 0.66, "years_needed": 1092.0, "se_at_now": 0.193,
         "realised_mean": 0.511, "implied_1pct": -0.055, "implied_2pct": -0.004,
         "implied_5pct": 0.083,
         "wf_cagr": 0.0774, "wf_base_cagr": 0.0712, "wf_min_w": 0.0, "wf_max_w": 0.20}
    h["best_weight"] = over.get("best_weight", 0.165)
    h.update(over)
    return h


def test_verdict_signal_reflects_the_in_sample_improvement():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(best_sharpe=0.70))["signal"] == "Weak"
    assert st.verdict(_headline(best_sharpe=0.70, best_cagr=0.05))["signal"] == "None"


def test_verdict_tradability_keys_off_the_gap_to_what_is_recommended():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(best_weight=0.06))["trad"] == "Fragile"
    assert st.verdict(_headline(best_weight=0.03))["trad"] == "Investable"


def test_verdict_prose_states_the_inversion_not_a_plateau():
    v = st.verdict(_headline())
    assert "not 1%, not 2%" in v["signal_why"]
    assert "365 days a year" in v["signal_why"]
    assert "after discarding it" in v["trad_why"]
    assert "standard error" in v["trad_why"]
    assert "override" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
