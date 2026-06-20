"""The engine's invariants and the study's three spines, all on the deterministic synthetic
tape (never the network): (1) the income illusion — a fat distribution sitting on an eroding
NAV reads as ~100% return of capital; (2) the energy beta — the fund regresses on the energy
factor with a HAC t above the bar (the Signal); (3) the null world has zero beta, no spread,
no illusion."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp_pipelines import strategy as st  # noqa: E402


# ---- performance primitives ------------------------------------------------
def test_cagr_of_constant_return():
    r = pd.Series([0.01] * 24)
    assert abs(st.cagr(r) - ((1.01 ** 12) - 1.0)) < 1e-9


def test_max_drawdown_is_nonpositive():
    r = pd.Series([0.05, -0.10, 0.02, -0.20, 0.03])
    assert st.max_drawdown(r) <= 0.0


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(st.sharpe(pd.Series([0.01] * 10)))


# ---- HAC / bootstrap inference ---------------------------------------------
def test_newey_west_recovers_mean():
    rng = np.random.default_rng(0)
    x = 0.002 + 0.01 * rng.standard_normal(400)
    mu, t = st.newey_west_t(x)
    assert abs(mu - x.mean()) < 1e-12
    assert t > 2.0


def test_newey_west_zero_mean_is_insignificant():
    rng = np.random.default_rng(0)
    x = 0.01 * rng.standard_normal(400)
    _, t = st.newey_west_t(x)
    assert abs(t) < 2.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(2)
    x = 0.003 + 0.01 * rng.standard_normal(300)
    lo, hi = st.block_bootstrap_ci(x, block=6, seed=42)
    assert lo < x.mean() < hi


# ---- the income illusion (spine #1) ----------------------------------------
def test_income_illusion_is_all_return_of_capital(trap):
    """A fat distribution on an eroding NAV → return-of-capital share = 1 (the trap)."""
    panel, _ = trap
    ill = st.income_illusion(panel["mlp_price"], panel["mlp_dist"])
    assert ill["dist_yield"] > 0.05          # marketed as high "income"
    assert ill["price_cagr"] < 0.0           # but the NAV is eroding
    assert ill["return_of_capital_share"] > 0.95   # so the income is your own capital


# ---- the energy beta (spine #2, the Signal) --------------------------------
def test_energy_beta_recovers_the_planted_slope(trap):
    """The decisive identity: the fund regresses on the energy factor with the planted beta and
    a HAC t that clears the inference bar (it's a leveraged energy bet, not bond-like income)."""
    panel, truth = trap
    eb = st.energy_beta(panel["mlp_tr"], panel["energy_tr"])
    # planted beta ~1.10 on the log factor; the simple-return regression recovers ~that
    assert eb["beta"] > 0.8
    assert eb["beta_t"] > 2.0                 # the energy exposure is statistically real
    assert eb["ci_lo"] > 0.0                  # the whole CI is above zero
    assert eb["r2"] > 0.5                     # energy explains most of the variance


def test_energy_capture_crashes_with_energy(trap):
    """A leveraged energy bet keeps a lot of BOTH up and down energy months (high down-capture
    — the opposite of a defensive 'income' sleeve)."""
    panel, _ = trap
    cap = st.capture(panel["mlp_price"], panel["energy_tr"])
    assert cap["down_capture"] > 0.8         # it falls hard when energy falls


# ---- the null world (spine #3) ---------------------------------------------
def test_null_world_has_no_energy_beta(null):
    """Beta 0, no drift → no energy exposure, no distribution illusion."""
    panel, _ = null
    eb = st.energy_beta(panel["mlp_tr"], panel["energy_tr"])
    assert abs(eb["beta"]) < 1e-6
    ill = st.income_illusion(panel["mlp_price"], panel["mlp_dist"])
    assert np.isnan(ill["return_of_capital_share"])  # no distribution → undefined ROC


# ---- the race --------------------------------------------------------------
def test_race_reports_spread_and_inference(trap):
    panel, _ = trap
    r = st.race(panel["mlp_tr"], panel["energy_tr"])
    assert r["n"] == len(panel)
    assert np.isfinite(r["spread_t"])
    assert np.isfinite(r["spread_ci_lo_bps"])


# ---- replicator invariants & costs -----------------------------------------
def test_replicate_identity_and_costs(trap):
    panel, _ = trap
    rep = st.replicate_mlp(panel["energy_tr"], beta=1.1, dist=0.0065, nav_drift=-0.0035, cost_bps=0.0)
    assert np.allclose(rep["total_gross"], rep["price"] + rep["dist"])
    rep2 = st.replicate_mlp(panel["energy_tr"], beta=1.1, dist=0.0065, cost_bps=5.0)
    assert np.allclose(rep2["total_net"], rep2["total_gross"] - 5.0e-4)


def test_cost_monotonically_lowers_net(trap):
    panel, _ = trap
    means = [
        st.replicate_mlp(panel["energy_tr"], beta=1.1, dist=0.0065, cost_bps=c)["total_net"].mean()
        for c in (0.0, 2.0, 5.0)
    ]
    assert means[0] > means[1] > means[2]


def test_execution_lag_changes_the_path_but_keeps_identity(trap):
    """The documented one-month execution-lag knob shifts when the exposure binds; identity holds."""
    panel, _ = trap
    rep0 = st.replicate_mlp(panel["energy_tr"], beta=1.1, dist=0.0065, lag=0)
    rep1 = st.replicate_mlp(panel["energy_tr"], beta=1.1, dist=0.0065, lag=1)
    assert not np.allclose(rep0["price"].to_numpy(), rep1["price"].to_numpy())
    assert np.allclose(rep1["total_gross"], rep1["price"] + rep1["dist"])
