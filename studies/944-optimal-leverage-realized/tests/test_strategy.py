"""Engine invariants, inference primitives, and the study's spine — all offline/synthetic.

The spine has two halves. **Recovery:** on a tape whose growth-optimal leverage is
planted at 2.0 the sweep must find its peak there and the Kelly estimator must return
2.0 — on average across seeds, and *conditionally* (argmax vs in-sample ``mu/sigma^2``)
on any single tape. **Quiet on the null:** with zero excess drift every unit of leverage
is pure variance drag, so the optimum must collapse onto the grid floor and the Kelly
estimator must centre on zero.

The single-seed scatter these tests tolerate is not sloppiness — it is the study's
finding in miniature: even 40 years of a *stationary, known* i.i.d. world does not pin
the growth-optimal multiple to better than about +/- 1.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimal_leverage import data, strategy as st  # noqa: E402

# The synthetic grid starts at 0.0 (the real-tape grid starts at 1.0) so the null has
# somewhere to collapse to — otherwise "the optimum is at the floor" is untestable.
SYN_GRID = np.round(np.arange(0.0, 3.0001, 0.25), 4)


# --------------------------------------------------------------------------- #
# Leg construction
# --------------------------------------------------------------------------- #
def test_prepare_synth_shape(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    assert {"r_asset", "r_cash", "acc"}.issubset(legs.columns)
    assert not legs.isna().any().any()
    assert len(legs) == len(prices) - 1


# --------------------------------------------------------------------------- #
# The daily-reset engine
# --------------------------------------------------------------------------- #
def test_lev_one_is_exactly_buy_and_hold(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    bt = st.levered_returns(legs, 1.0, spread_bps=200.0, cost_bps=25.0)
    # At L = 1 there is nothing borrowed and nothing to reset, whatever the assumptions.
    assert np.allclose(bt["r_lev"].to_numpy(), legs["r_asset"].to_numpy())
    assert float(bt["turnover"].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_excess_return_scales_linearly_gross_of_the_spread(planted):
    """Gross of financing spread and cost, ``e_lev`` is exactly ``L * (r_asset - r_cash)``."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    base = (legs["r_asset"] - legs["r_cash"]).to_numpy()
    for lev in (1.5, 2.0, 3.0):
        bt = st.levered_returns(legs, lev, spread_bps=0.0, cost_bps=0.0)
        assert np.allclose(bt["e_lev"].to_numpy(), lev * base)


def test_sharpe_is_invariant_in_leverage_gross_of_the_spread(planted):
    """The study's structural fact: constant leverage cannot move the Sharpe ratio."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    tab = st.sweep(legs, grid=(1.0, 1.5, 2.0, 3.0), spread_bps=0.0, cost_bps=0.0)
    assert tab["excess_sharpe"].std(ddof=0) < 1e-9


def test_spread_and_cost_only_ever_reduce_growth(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    g = [float(st.sweep(legs, grid=(2.0,), spread_bps=s, cost_bps=0.0)["log_growth_ann"].iloc[0])
         for s in (0.0, 50.0, 200.0)]
    assert g[0] > g[1] > g[2]
    c = [float(st.sweep(legs, grid=(2.0,), spread_bps=0.0, cost_bps=k)["log_growth_ann"].iloc[0])
         for k in (0.0, 1.0, 10.0)]
    assert c[0] > c[1] > c[2]


def test_growth_curve_is_concave_and_turnover_grows_with_leverage(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    tab = st.sweep(legs, grid=np.round(np.arange(0.0, 4.0001, 0.25), 4),
                   spread_bps=0.0, cost_bps=0.0)
    g = tab["log_growth_ann"].to_numpy()
    # Concave: second differences of a quadratic-in-L growth curve are negative.
    assert (np.diff(g, 2) < 0).all()
    # Reset turnover is V-shaped around L = 1 (a de-levered sleeve must be reset too);
    # it is exactly zero at L = 1 and rises monotonically above it.
    turn = pd.Series(tab["turnover_ann"])
    assert turn.loc[1.0] == pytest.approx(0.0, abs=1e-9)
    assert turn.loc[1.0] == turn.min()
    above = turn.loc[1.0:].to_numpy()
    assert (np.diff(above) > 0).all()


def test_returns_are_floored_at_total_loss():
    """A −60% day at 3x must wipe the vehicle out, not send its NAV negative."""
    idx = pd.bdate_range("2020-01-02", periods=3)
    legs = pd.DataFrame({"r_asset": [0.0, -0.60, 0.0], "r_cash": 0.0, "acc": 1 / 252.0},
                        index=idx)
    bt = st.levered_returns(legs, 3.0, spread_bps=0.0, cost_bps=0.0)
    assert float(bt["r_lev"].min()) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Kelly / theory
# --------------------------------------------------------------------------- #
def test_kelly_leverage_matches_the_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0004, 0.01, 20000)
    assert st.kelly_leverage(x) == pytest.approx(x.mean() / x.var(ddof=1), rel=1e-9)


def test_kelly_is_zero_when_the_asset_earns_only_cash():
    idx = pd.bdate_range("2010-01-04", periods=1000)
    legs = pd.DataFrame({"r_asset": 0.0002, "r_cash": 0.0002, "acc": 1 / 252.0}, index=idx)
    assert st.kelly_from_legs(legs) == pytest.approx(0.0) or np.isnan(st.kelly_from_legs(legs))


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_block_bootstrap_ci_brackets_the_point():
    rng = np.random.default_rng(2)
    ci = st.block_bootstrap_ci(rng.normal(0.001, 0.01, 3000), n_boot=300, seed=944)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_max_drawdown_is_negative_on_a_falling_path():
    r = pd.Series([0.1, -0.5, 0.1])
    assert st.max_drawdown(r) < -0.4


# --------------------------------------------------------------------------- #
# The one execution lag
# --------------------------------------------------------------------------- #
def test_ex_ante_kelly_is_lagged_and_clipped(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    out = st.ex_ante_kelly(legs, window=252, lo=1.0, hi=3.0)
    assert out["lev"].between(1.0, 3.0).all()
    # Rebuild the unlagged estimate and check the applied multiple is yesterday's.
    e = legs["r_asset"] - legs["r_cash"]
    raw = (e.rolling(252, min_periods=252).mean()
           / e.rolling(252, min_periods=252).var(ddof=1)).clip(1.0, 3.0)
    assert np.allclose(out["lev"].to_numpy(), raw.shift(1).dropna().to_numpy())


def test_ex_ante_kelly_ignores_the_future(planted):
    """Perturbing only the tail of the tape must not change any earlier applied multiple."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    a = st.ex_ante_kelly(legs, window=252)
    legs2 = legs.copy()
    legs2.iloc[-500:, legs2.columns.get_loc("r_asset")] += 0.05
    b = st.ex_ante_kelly(legs2, window=252)
    n = len(a) - 500
    assert np.allclose(a["lev"].to_numpy()[:n], b["lev"].to_numpy()[:n])


# --------------------------------------------------------------------------- #
# The spine (1) — recovery of the planted optimum
# --------------------------------------------------------------------------- #
def test_argmax_tracks_the_in_sample_kelly_on_any_single_tape(planted):
    """Conditional consistency: the realised peak sits at the tape's own mu/sigma^2."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    fine = np.round(np.arange(0.0, 5.0001, 0.05), 4)
    opt = st.realised_optimum(legs, grid=fine, spread_bps=0.0, cost_bps=0.0)
    assert abs(opt - st.kelly_from_legs(legs)) < 0.3


def test_planted_optimum_is_recovered_on_average():
    opts = [st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=944 + s)[0],
                                grid=SYN_GRID)["opt_lev"] for s in range(8)]
    assert abs(float(np.mean(opts)) - 2.0) < 0.8
    assert float(np.mean(opts)) > 1.2


def test_planted_kelly_estimator_is_unbiased_on_average():
    ks = [st.kelly_from_legs(st.prepare_synth(
        data.synthetic_daily(signal_strength=1.0, seed=944 + s)[0])) for s in range(24)]
    assert abs(float(np.mean(ks)) - 2.0) < 0.7


# --------------------------------------------------------------------------- #
# The spine (2) — quiet on the null
# --------------------------------------------------------------------------- #
def test_null_optimum_collapses_to_the_grid_floor():
    opts = [st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=944 + s)[0],
                                grid=SYN_GRID)["opt_lev"] for s in range(8)]
    assert float(np.mean(opts)) < 0.9
    assert float(np.median(opts)) <= 0.5


def test_null_kelly_estimator_is_centred_on_zero():
    ks = [st.kelly_from_legs(st.prepare_synth(
        data.synthetic_daily(signal_strength=0.0, seed=944 + s)[0])) for s in range(24)]
    assert abs(float(np.mean(ks))) < 0.6


def test_planted_and_null_are_cleanly_separated():
    def mean_opt(ss):
        return float(np.mean([
            st.synthetic_detect(data.synthetic_daily(signal_strength=ss, seed=944 + s)[0],
                                grid=SYN_GRID)["opt_lev"] for s in range(8)]))
    assert mean_opt(1.0) - mean_opt(0.0) > 1.0


# --------------------------------------------------------------------------- #
# Instability machinery
# --------------------------------------------------------------------------- #
def test_rolling_optimum_and_instability_shapes(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    roll = st.rolling_optimum(legs, grid=SYN_GRID, window=1260, step=252,
                              spread_bps=0.0, cost_bps=0.0)
    assert {"opt_lev", "kelly", "vol_ann", "exret_ann"}.issubset(roll.columns)
    assert len(roll) > 5
    inst = st.instability(roll, grid=SYN_GRID)
    assert inst["min"] <= inst["mean"] <= inst["max"]
    assert 0.0 <= inst["frac_at_cap"] <= 1.0


def test_bootstrap_optimum_ci_brackets_the_point(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    b = st.bootstrap_optimum(legs, grid=SYN_GRID, n_boot=60, block=63,
                             spread_bps=0.0, cost_bps=0.0, seed=944)
    assert b["ci_low"] <= b["opt"] <= b["ci_high"]
    assert b["sd"] >= 0.0


def test_era_cut_returns_both_halves(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    eras = st.era_cut(legs, split="2006-01-01", grid=SYN_GRID, spread_bps=0.0, cost_bps=0.0)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["opt_lev"]) and np.isfinite(e["kelly"])


def test_growth_diff_test_is_zero_against_itself(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    a = st.levered_returns(legs, 2.0)
    d = st.growth_diff_test(a, a, seed=944)
    assert d["log_growth_diff_ann"] == pytest.approx(0.0, abs=1e-12)


def test_growth_diff_test_detects_a_planted_growth_gap(planted):
    """On the planted tape, L = 2 must compound faster than L = 1 (that is the DGP)."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    d = st.growth_diff_test(st.levered_returns(legs, 2.0, spread_bps=0.0, cost_bps=0.0),
                            st.levered_returns(legs, 1.0), seed=944)
    assert d["log_growth_diff_ann"] > 0.0


def test_sweeps_return_the_expected_columns(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    s = st.spread_sweep(legs, grid=SYN_GRID, spreads=(0.0, 100.0))
    c = st.cost_sweep(legs, grid=SYN_GRID, costs=(0.0, 5.0))
    assert {"opt_lev", "cagr_opt", "sharpe_opt"}.issubset(s.columns)
    assert {"opt_lev", "turnover_ann_opt"}.issubset(c.columns)
    # A wider spread can never raise the optimum.
    assert s.loc[100.0, "opt_lev"] <= s.loc[0.0, "opt_lev"]


def test_sharpe_diff_test_is_zero_against_itself(planted):
    """Constant leverage cannot move Sharpe, so an arm raced against itself must give 0."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    a = st.levered_returns(legs, 2.0, spread_bps=0.0, cost_bps=0.0)
    d = st.sharpe_diff_test(a, a, n_boot=40, seed=944)
    assert d["diff"] == pytest.approx(0.0, abs=1e-12)
    assert d["ci_low"] <= 0.0 <= d["ci_high"]


def test_sharpe_is_invariant_in_leverage_gross_of_costs(planted):
    """The construction check the study leans on: L only scales the excess stream."""
    prices, _ = planted
    legs = st.prepare_synth(prices)
    d = st.sharpe_diff_test(st.levered_returns(legs, 3.0, spread_bps=0.0, cost_bps=0.0),
                            st.levered_returns(legs, 1.0, spread_bps=0.0, cost_bps=0.0),
                            n_boot=40, seed=944)
    assert d["diff"] == pytest.approx(0.0, abs=1e-9)


def test_start_sensitivity_moves_nothing_on_a_stationary_tape(planted):
    """On an i.i.d. tape the optimum should be *roughly* start-invariant.

    The real-tape version of this test is the study's sharpest finding (the optimum and
    the sign of the era hand-off both move with the window's left edge); here we only
    assert the machinery runs and that a stationary DGP does not produce the wild swings
    the real tape does.
    """
    prices, _ = planted
    legs = st.prepare_synth(prices)
    starts = [str(legs.index[0].date()), str(legs.index[500].date()),
              str(legs.index[1000].date())]
    ss = st.start_sensitivity(legs, starts=starts, grid=SYN_GRID, split="2006-01-01",
                              spread_bps=0.0, cost_bps=0.0)
    assert len(ss) == 3
    assert {"opt_lev", "handoff_edge", "late_opt"}.issubset(ss.columns)
    assert ss["opt_lev"].max() - ss["opt_lev"].min() <= 1.0


def test_start_sensitivity_skips_windows_too_short_to_split(planted):
    prices, _ = planted
    legs = st.prepare_synth(prices)
    ss = st.start_sensitivity(legs, starts=("2005-06-01",), grid=SYN_GRID,
                              split="2005-07-01", spread_bps=0.0, cost_bps=0.0)
    assert len(ss) == 0
