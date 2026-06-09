"""The crux: the two-sided HP trend depends on FUTURE data (the trap); the one-sided causal trend does
not. We prove it directly by perturbing a future point and watching which trend moves."""

import numpy as np
import pandas as pd

from crystal_ball import data, hp


def test_two_sided_uses_the_future(null_close):
    """Perturb a point 200 days ahead; the two-sided trend at time t MUST move (it sees the future)."""
    yl = np.log(null_close.astype(float))
    t = 1000
    tau_a = hp.hp_trend_twosided(yl, lam=1e6).iloc[t]
    yl2 = yl.copy(); yl2.iloc[t + 200] += 0.5          # big bump well after t
    tau_b = hp.hp_trend_twosided(yl2, lam=1e6).iloc[t]
    assert abs(tau_b - tau_a) > 1e-6                    # the future changed the past estimate -> look-ahead


def test_one_sided_ignores_the_future(null_close):
    """The causal trend at t uses only y[t-window:t+1]; a future bump must leave it UNCHANGED."""
    yl = np.log(null_close.astype(float))
    t = 1000
    tau_a = hp.hp_trend_onesided(yl, lam=1e6, window=252).iloc[t]
    yl2 = yl.copy(); yl2.iloc[t + 200] += 0.5
    tau_b = hp.hp_trend_onesided(yl2, lam=1e6, window=252).iloc[t]
    assert abs(tau_b - tau_a) < 1e-9                    # causal: the future cannot touch it


def test_cycle_centered_and_shaped(revert_close):
    c2 = hp.cycle(revert_close, lam=1e6, causal=False)
    c1 = hp.cycle(revert_close, lam=1e6, causal=True, window=252)
    assert abs(c2.mean()) < 0.05                        # cycle is a deviation from trend (~centered)
    assert c1.iloc[:252].isna().all()                  # causal warm-up is NaN
    assert c1.iloc[252:].notna().any()


def test_deterministic(revert):
    close, truth = revert
    close2, _ = data.synthetic_prices(revert_rho=0.97, seed=22)
    assert np.allclose(close.to_numpy(), close2.to_numpy())
    assert truth.has_reversion
