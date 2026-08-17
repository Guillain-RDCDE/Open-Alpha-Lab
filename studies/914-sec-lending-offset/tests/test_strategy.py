"""Estimator logic, trade invariants, and the study's spine — all offline/synthetic.

The spine: on a pair with a *planted* lending passthrough the fee-adjusted residual
estimator recovers the planted basis points with |t| >= 2; on the null (fees and
composition noise only) it is centred on zero and does not fire. The pair trade charges
borrow monotonically, resets with one execution lag, and the expense-ratio assumption
moves the answer one-for-one (so the sweep is honest about what is tape and what is not).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lending_offset import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_finite():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0, 1, 400), rng.normal(0.2, 1, 400)
    assert np.isfinite(st.one_sample_t(a))
    assert np.isfinite(st.welch_t(a, b))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(3)
    x = 0.0005 + rng.normal(0, 0.004, 600)
    ci = st.block_bootstrap_mean_ci(x, n_boot=800, block=6, seed=914)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# Return construction
# --------------------------------------------------------------------------- #
def test_monthly_returns_are_month_end_and_complete(planted_pair):
    prices, _ = planted_pair
    m = st.monthly_log_returns(prices)
    assert not m.isna().any().any()
    assert m.index.is_monotonic_increasing
    assert len(m) < len(prices)


def test_log_return_difference_is_scale_invariant(planted_pair):
    """Rescaling one fund's price level must not move the residual (it is a ratio)."""
    prices, truth = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    base = st.pair_residual(px, "A", "B", truth["er_a"], truth["er_b"], n_boot=1)
    px2 = px.copy()
    px2["A"] = px2["A"] * 7.5
    scaled = st.pair_residual(px2, "A", "B", truth["er_a"], truth["er_b"], n_boot=1)
    assert scaled["residual_bp"] == pytest.approx(base["residual_bp"], abs=1e-6)


# --------------------------------------------------------------------------- #
# The estimator's arithmetic
# --------------------------------------------------------------------------- #
def test_residual_equals_drift_plus_fee_gap(planted_pair):
    prices, truth = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    r = st.pair_residual(px, "A", "B", truth["er_a"], truth["er_b"], n_boot=1)
    assert r["residual_bp"] == pytest.approx(r["drift_bp"] + r["delta_er_bp"], abs=1e-6)
    assert r["delta_er_bp"] == pytest.approx((truth["er_a"] - truth["er_b"]) * 100.0)


def test_identical_funds_give_zero_residual():
    """Two copies of the same tape with the same fee must produce an exactly zero gap."""
    prices, _ = data.synthetic_daily(seed=914)
    px = pd.DataFrame({"A": prices["fund_a"], "B": prices["fund_a"]})
    r = st.pair_residual(px, "A", "B", 0.20, 0.20, n_boot=1)
    assert abs(r["residual_bp"]) < 1e-6
    assert r["te_pct"] == pytest.approx(0.0, abs=1e-9)


def test_er_assumption_moves_residual_one_for_one(planted_pair):
    prices, truth = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    rows = st.er_sweep(px, "A", "B", (0.50, 0.70, 0.90), truth["er_b"])
    # +20 bp of assumed fee on leg A must add exactly +20 bp to the residual.
    assert rows[1]["residual_bp"] - rows[0]["residual_bp"] == pytest.approx(20.0, abs=1e-6)
    assert rows[2]["residual_bp"] - rows[1]["residual_bp"] == pytest.approx(20.0, abs=1e-6)


def test_cheap_leg_er_moves_residual_one_for_one_downward(planted_pair):
    """Raising the CHEAP leg's assumed fee lowers the residual one-for-one.

    This is the sweep that decides the residual's sign on the real tape: every cheap leg
    had its fee cut during the sample, so pinning it at today's value biases the residual
    UP. The estimator must respond to that assumption exactly, and in this direction.
    """
    prices, truth = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    rows = st.er_sweep_b(px, "A", "B", truth["er_a"], (0.05, 0.10, 0.15))
    assert rows[1]["residual_bp"] - rows[0]["residual_bp"] == pytest.approx(-5.0, abs=1e-6)
    assert rows[2]["residual_bp"] - rows[1]["residual_bp"] == pytest.approx(-5.0, abs=1e-6)
    # ...and it is the mirror image of sweeping the dear leg.
    up = st.er_sweep(px, "A", "B", (truth["er_a"], truth["er_a"] + 0.05), 0.10)
    assert up[1]["residual_bp"] - up[0]["residual_bp"] == pytest.approx(+5.0, abs=1e-6)


def test_noise_floor_shrinks_with_sample_length():
    """The smallest resolvable effect must fall as the sample grows (root-n)."""
    short_px, tr = data.synthetic_daily(n_years=6, seed=914)
    long_px, _ = data.synthetic_daily(n_years=25, seed=914)
    rs = st.pair_residual(short_px.rename(columns={"fund_a": "A", "fund_b": "B"}),
                          "A", "B", tr["er_a"], tr["er_b"], n_boot=1)
    rl = st.pair_residual(long_px.rename(columns={"fund_a": "A", "fund_b": "B"}),
                          "A", "B", tr["er_a"], tr["er_b"], n_boot=1)
    assert rl["noise_floor_bp"] < rs["noise_floor_bp"]


# --------------------------------------------------------------------------- #
# Pair trade invariants
# --------------------------------------------------------------------------- #
def test_borrow_monotonically_reduces_the_pair_return(planted_pair):
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    means = [st.pair_trade(px, "B", "A", borrow_bps=bb).mean() for bb in (0.0, 25.0, 100.0)]
    assert means[0] > means[1] > means[2]


def test_costs_monotonically_reduce_the_pair_return(planted_pair):
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    means = [st.pair_trade(px, "B", "A", borrow_bps=0.0, cost_bps=c).mean()
             for c in (0.0, 3.0, 20.0)]
    assert means[0] >= means[1] >= means[2]


def test_pair_trade_friction_is_booked_after_the_book_date(planted_pair):
    """Friction lands in January, one period after the December book date.

    This is a **cost convention**, not an execution lag: the pair is static (two fixed
    tickers, no signal), so there is no decision to lag and the study claims none.
    """
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    free = st.pair_trade(px, "B", "A", borrow_bps=0.0, cost_bps=0.0)
    costed = st.pair_trade(px, "B", "A", borrow_bps=0.0, cost_bps=10.0)
    charged = (free - costed).round(12)
    hit = charged[charged > 0]
    assert len(hit) > 3
    assert set(hit.index.month) == {1}


def test_pair_trade_is_antisymmetric(planted_pair):
    """Swapping the legs must flip the gross sign (borrow and cost aside)."""
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    ab = st.pair_trade(px, "A", "B", borrow_bps=0.0, cost_bps=0.0)
    ba = st.pair_trade(px, "B", "A", borrow_bps=0.0, cost_bps=0.0)
    assert np.allclose(ab.to_numpy(), -ba.to_numpy())


def test_charged_friction_exceeds_true_rebalancing_turnover(planted_pair):
    """The annual full-notional charge must be CONSERVATIVE, not flattering.

    Restoring equal notional only trades the legs' drift, so the true cost is tiny; the
    study charges far more. If this ever inverts, the pair trade is being flattered by its
    own cost model and the claim in the docstring is false.
    """
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    t = st.rebalance_turnover(px, "B", "A", cost_bps=3.0)
    assert t["charged_bp_yr"] == pytest.approx(6.0)
    assert 0.0 < t["true_cost_bp_yr"] < t["charged_bp_yr"]
    assert t["turnover_ann_pct"] > 0.0


def test_borrow_sweep_shape(planted_pair):
    prices, _ = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    rows = st.borrow_sweep(px, "B", "A", borrow_grid=(0.0, 50.0))
    assert len(rows) == 2
    for r in rows:
        assert {"ann_bp", "sharpe", "t_hac", "max_drawdown", "borrow_bps"}.issubset(r)
        assert np.isfinite(r["sharpe"])


# --------------------------------------------------------------------------- #
# The study's spine — the estimator is unbiased
# --------------------------------------------------------------------------- #
def test_planted_passthrough_is_recovered(planted_pair):
    prices, truth = planted_pair
    d = st.synthetic_detect(prices, truth)
    assert abs(d["error_bp"]) < 40.0          # within ~2 standard errors of +60 bp
    assert d["t_hac"] > 2.0                   # and it clears the desk's inference bar


def test_null_stays_quiet(null_pair):
    prices, truth = null_pair
    d = st.synthetic_detect(prices, truth)
    assert abs(d["residual_bp"]) < 40.0
    assert abs(d["t_hac"]) < 2.0


def test_null_across_seeds_is_centred():
    resids, ts = [], []
    for s in range(8):
        px, tr = data.synthetic_daily(signal_strength=0.0, seed=914 + s)
        d = st.synthetic_detect(px, tr)
        resids.append(d["residual_bp"]); ts.append(d["t_hac"])
    resids, ts = np.array(resids), np.array(ts)
    assert abs(resids.mean()) < 20.0
    assert (np.abs(ts) >= 2.0).sum() <= 1      # a 5% test may fire once in eight


def test_half_strength_lands_between(planted_pair, null_pair):
    px, tr = data.synthetic_daily(signal_strength=0.5, seed=914)
    d_half = st.synthetic_detect(px, tr)
    d_full = st.synthetic_detect(*planted_pair)
    d_null = st.synthetic_detect(*null_pair)
    assert d_null["residual_bp"] < d_half["residual_bp"] < d_full["residual_bp"]


def test_pooling_sharpens_the_estimate(planted_panel):
    frames, truth = planted_panel
    pooled = st.synthetic_panel_detect(frames, truth)
    single = st.synthetic_detect(*data.synthetic_daily(signal_strength=1.0, seed=914))
    assert abs(pooled["residual_bp"] - truth["planted_residual_bp"]) < 40.0
    assert pooled["t_hac"] > single["t_hac"]   # five pairs beat one


def test_daily_and_monthly_agree_on_the_planted_effect(planted_pair):
    prices, truth = planted_pair
    m = st.synthetic_detect(prices, truth, freq="M")
    d = st.synthetic_detect(prices, truth, freq="D")
    assert abs(m["residual_bp"] - d["residual_bp"]) < 5.0


def test_era_cut_returns_both_halves(planted_pair):
    prices, truth = planted_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    split = str(px.index[len(px) // 2].date())
    eras = st.era_cut(px, "A", "B", truth["er_a"], truth["er_b"], split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["residual_bp"]) and np.isfinite(e["t_hac"])


def test_upper_bound_is_a_magnitude(null_pair):
    prices, truth = null_pair
    px = prices.rename(columns={"fund_a": "A", "fund_b": "B"})
    r = st.pair_residual(px, "A", "B", truth["er_a"], truth["er_b"], n_boot=400)
    assert st.upper_bound_bp(r) >= 0.0
    assert st.upper_bound_bp(r, side="high") >= 0.0
