"""Strategy tests for Study 997 — timing luck, planted and removed."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lottery import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def test_rebalance_dates_are_spaced_by_the_period():
    idx = pd.bdate_range("2020-01-01", periods=200)
    d = st.rebalance_dates(idx, period=21, offset=0)
    positions = [idx.get_loc(x) for x in d]
    assert all(b - a == 21 for a, b in zip(positions, positions[1:]))


def test_the_offset_shifts_the_schedule():
    idx = pd.bdate_range("2020-01-01", periods=200)
    a = st.rebalance_dates(idx, 21, 0)
    b = st.rebalance_dates(idx, 21, 5)
    assert idx.get_loc(b[0]) - idx.get_loc(a[0]) == 5
    assert len(set(a) & set(b)) == 0


def test_offsets_beyond_the_period_wrap_around():
    idx = pd.bdate_range("2020-01-01", periods=200)
    assert list(st.rebalance_dates(idx, 21, 0)) == list(st.rebalance_dates(idx, 21, 21))


def test_a_zero_period_is_rejected():
    idx = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        st.rebalance_dates(idx, period=0)


def test_the_offsets_partition_the_sessions():
    idx = pd.bdate_range("2020-01-01", periods=210)
    seen = set()
    for off in range(21):
        seen |= set(st.rebalance_dates(idx, 21, off))
    assert len(seen) == len(idx)


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #
def test_the_fixed_weight_rule_always_returns_its_target():
    px = st.synthetic_prices(n=500, n_assets=3)
    rule = st.fixed_weight_rule(px, {"A0": 0.6, "A1": 0.4})
    assert rule(px.index[100], px.iloc[:100]) == {"A0": 0.6, "A1": 0.4}


def test_the_momentum_rule_picks_the_best_performers():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame({"A": np.linspace(100, 200, 300),      # best
                       "B": np.linspace(100, 150, 300),
                       "C": np.linspace(100, 90, 300)},      # worst
                      index=idx)
    rule = st.momentum_rule(lookback=126, n_hold=2)
    w = rule(idx[-1], px)
    assert set(w) == {"A", "B"}
    assert w["A"] == pytest.approx(0.5)


def test_the_momentum_rule_declines_without_enough_history():
    px = st.synthetic_prices(n=500, n_assets=4)
    assert st.momentum_rule(lookback=126)(px.index[10], px.iloc[:10]) == {}


def test_a_longer_lookback_changes_the_selection():
    px = st.synthetic_prices(n=1500, n_assets=8, momentum=1.0)
    short = st.momentum_rule(21, 3)(px.index[-1], px)
    long = st.momentum_rule(252, 3)(px.index[-1], px)
    assert set(short) != set(long)


# --------------------------------------------------------------------------- #
# Running a strategy
# --------------------------------------------------------------------------- #
def test_a_buy_and_hold_rule_tracks_its_asset():
    px = st.synthetic_prices(n=2000, n_assets=2)
    r = st.run_strategy(px, st.fixed_weight_rule(px, {"A0": 1.0}), period=21,
                        cost_bps=0.0)
    expected = float(px["A0"].iloc[-1] / px["A0"].iloc[0])
    assert r["final"] == pytest.approx(expected, rel=0.02)


def test_costs_reduce_the_final_value():
    px = st.synthetic_prices(n=2500, n_assets=6, momentum=1.0)
    free = st.run_strategy(px, st.momentum_rule(), 21, 0, cost_bps=0.0)
    paid = st.run_strategy(px, st.momentum_rule(), 21, 0, cost_bps=100.0)
    assert paid["final"] < free["final"]


def test_the_momentum_rule_actually_turns_over():
    px = st.synthetic_prices(n=2500, n_assets=8)
    r = st.run_strategy(px, st.momentum_rule(), 21, 0)
    assert r["turnover_per_year"] > 0.5


def test_a_fixed_weight_rule_turns_over_less_than_momentum():
    px = st.synthetic_prices(n=2500, n_assets=8)
    fw = st.run_strategy(px, st.fixed_weight_rule(px, {"A0": 0.6, "A1": 0.4}), 21, 0)
    mom = st.run_strategy(px, st.momentum_rule(), 21, 0)
    assert fw["turnover_per_year"] < mom["turnover_per_year"]


def test_run_strategy_declines_on_a_short_panel():
    px = st.synthetic_prices(n=40, n_assets=3)
    assert "cagr" not in st.run_strategy(px, st.momentum_rule(), 21, 0)


# --------------------------------------------------------------------------- #
# The luck itself
# --------------------------------------------------------------------------- #
def test_every_offset_is_run():
    px = st.synthetic_prices(n=2000, n_assets=6)
    v = st.run_variants(px, st.momentum_rule(), period=10)
    assert len(v) == 10
    assert list(v.index) == list(range(10))


def test_offsets_give_different_answers_for_a_ranking_rule():
    """The finding, in one assertion."""
    px = st.synthetic_prices(n=3000, n_assets=8)
    v = st.run_variants(px, st.momentum_rule(), period=21)
    assert v["cagr"].std(ddof=1) > 0.002
    assert v["cagr"].max() - v["cagr"].min() > 0.005


def test_a_fixed_weight_rule_has_far_less_timing_luck():
    """The control that isolates the mechanism: same assets means small dispersion."""
    px = st.synthetic_prices(n=3000, n_assets=8)
    fw = st.run_variants(px, st.fixed_weight_rule(px, {"A0": 0.6, "A1": 0.4}), 21)
    mom = st.run_variants(px, st.momentum_rule(), 21)
    assert (fw["cagr"].max() - fw["cagr"].min()) < (mom["cagr"].max() - mom["cagr"].min())


def test_timing_luck_exists_even_with_no_signal_at_all():
    """Under the null there is nothing to find, so all the dispersion is luck."""
    px = st.synthetic_prices(n=3000, n_assets=8, momentum=0.0)
    s = st.luck_summary(st.run_variants(px, st.momentum_rule(), 21))
    assert s["cagr_spread"] > 0.003
    assert s["final_ratio"] > 1.05


def test_luck_summary_reports_the_extremes():
    px = st.synthetic_prices(n=2500, n_assets=6)
    v = st.run_variants(px, st.momentum_rule(), 21)
    s = st.luck_summary(v)
    assert s["cagr_max"] == pytest.approx(v["cagr"].max())
    assert s["best_offset"] == v["cagr"].idxmax()
    assert s["final_ratio"] >= 1.0


def test_luck_summary_handles_an_empty_frame():
    assert st.luck_summary(pd.DataFrame()) == {}


def test_longer_rebalance_periods_carry_more_luck():
    px = st.synthetic_prices(n=4000, n_assets=8)
    sw = st.period_sweep(px, lambda: st.momentum_rule(), periods=(5, 21, 63))
    assert sw.loc[63, "cagr_spread"] > sw.loc[5, "cagr_spread"]


def test_luck_vs_signal_flags_when_the_luck_wins():
    px = st.synthetic_prices(n=3000, n_assets=8, momentum=0.0)
    v = st.run_variants(px, st.momentum_rule(), 21)
    bench = float((px.mean(axis=1).iloc[-1] / px.mean(axis=1).iloc[0])
                  ** (252 / len(px)) - 1)
    out = st.luck_vs_signal(v, bench)
    assert out["swamped"]
    assert 0.0 <= out["share_beating_benchmark"] <= 1.0


def test_luck_vs_signal_handles_an_empty_frame():
    assert st.luck_vs_signal(pd.DataFrame(), 0.05) == {}


# --------------------------------------------------------------------------- #
# The fix
# --------------------------------------------------------------------------- #
def test_overlapping_portfolios_sit_inside_the_variant_range():
    px = st.synthetic_prices(n=3000, n_assets=8)
    v = st.run_variants(px, st.momentum_rule(), 21)
    o = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    assert v["cagr"].min() <= o["cagr"] <= v["cagr"].max()


def test_overlapping_portfolios_reduce_volatility():
    """Because the sleeves are imperfectly correlated with each other."""
    px = st.synthetic_prices(n=3000, n_assets=8)
    o = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    assert o["vol_reduction"] > 0


def test_overlapping_portfolios_improve_the_average_drawdown():
    px = st.synthetic_prices(n=4000, n_assets=8)
    o = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    assert o["max_dd"] > o["mean_variant_dd"]      # less negative is better


def test_the_fix_preserves_a_genuine_signal():
    """A fix that removed the signal along with the noise would be worthless."""
    px = st.synthetic_prices(n=4000, n_assets=8, momentum=2.0)
    o = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    assert o["sharpe"] >= o["mean_variant_sharpe"] - 0.05


def test_overlapping_portfolios_have_no_timing_luck_by_construction():
    """There is only one blended portfolio, so there is nothing left to vary."""
    px = st.synthetic_prices(n=2500, n_assets=6)
    a = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    b = st.overlapping_portfolios(px, st.momentum_rule(), 21)
    assert a["cagr"] == pytest.approx(b["cagr"])


def test_overlapping_portfolios_decline_on_a_short_panel():
    px = st.synthetic_prices(n=40, n_assets=3)
    assert st.overlapping_portfolios(px, st.momentum_rule(), 21) == {}


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_offsets": 21, "years": 22.0, "mom_cagr_min": 0.051, "mom_cagr_max": 0.089,
         "mom_cagr_spread": 0.038, "mom_final_ratio": 2.11, "mom_edge": 0.014,
         "mom_spread_over_edge": 2.7, "mom_share_beating": 0.62,
         "fw_cagr_spread": 0.004, "blend_cagr": 0.071, "blend_sharpe": 0.58,
         "mean_variant_cagr": 0.069, "mean_variant_sharpe": 0.54,
         "vol_reduction": 0.008, "dd_improvement": 0.031}
    h.update(over)
    return h


def test_verdict_signal_needs_the_luck_to_swamp_the_edge():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(mom_spread_over_edge=0.4))["signal"] == "Partial"
    assert st.verdict(_headline(mom_cagr_spread=0.002))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(blend_sharpe=0.50))["trad"] == "Partial"
    assert st.verdict(_headline(blend_sharpe=0.30))["trad"] == "Mirage"


def test_verdict_prose_contrasts_the_two_rule_families():
    v = st.verdict(_headline())
    assert "fixed-weight" in v["signal_why"] and "ranking rule" in v["signal_why"]
    assert "overlapping" in v["trad_why"].lower() or "all 21 offsets" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
