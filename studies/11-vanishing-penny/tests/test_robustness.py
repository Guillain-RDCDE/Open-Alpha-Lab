"""The honesty layer — survival shape, bootstrap CI, the resolution caveat, retail capture."""

import numpy as np

from prediction_arb import arbitrage, robustness


def test_survival_curve_is_monotone(gap):
    eps = arbitrage.detect_all(gap)
    surv = robustness.survival_curve(eps)
    f = surv["frac_open"].to_numpy()
    assert f[0] == 1.0 or np.isclose(f[0], 1.0)
    assert np.all(np.diff(f) <= 1e-9)                 # non-increasing
    assert f[-1] <= 0.05                              # nearly everything has closed


def test_bootstrap_ci_brackets_the_point(gap, truth):
    eps = arbitrage.detect_all(eps_gap := gap)
    b = robustness.bootstrap_half_life(eps, n_boot=500, seed=0)
    assert b["ci_low"] <= b["half_life_min"] <= b["ci_high"]
    assert b["ci_high"] - b["ci_low"] < 4.0           # a tight CI on a clean synthetic
    assert abs(b["half_life_min"] - truth.half_life_min) <= 1.2


def test_coarser_tape_sees_fewer_episodes(gap):
    """The load-bearing caveat: coarsening the tape destroys fast episodes."""
    sweep = robustness.resolution_sweep(gap, fidelities=(1, 5, 30, 60))
    seen = sweep["frac_episodes_seen"].to_numpy()
    assert seen[0] == 1.0
    assert seen[-1] < seen[0]                          # the 60-min tape sees far fewer
    # the half-life you 'measure' inflates as the grid coarsens past the true 6 min
    assert sweep["half_life_min"].iloc[-1] >= sweep["half_life_min"].iloc[0]


def test_retail_capture_is_brutal_and_exact():
    # at one half-life of latency, exactly half the penny is gone
    assert np.isclose(robustness.retail_capture(6.0, 6.0), 0.5)
    # monotone decreasing in latency
    tbl = robustness.retail_capture_table(6.0)
    caps = tbl["capture_frac"].to_numpy()
    assert np.all(np.diff(caps) <= 0)
    # a 30-ms bot keeps essentially all of it; a 30-min human keeps almost none
    assert robustness.retail_capture(6.0, 0.0005) > 0.99
    assert robustness.retail_capture(6.0, 30.0) < 0.05
