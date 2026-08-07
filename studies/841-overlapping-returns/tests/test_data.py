"""The synthetic monthly world is deterministic, offline, and carries the planted truth it claims:
zero predictability under the null, a genuine edge under the control, and a persistent predictor."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlapping_returns import data  # noqa: E402


def test_world_deterministic():
    a, _ = data.simulate_world(seed=841)
    b, _ = data.simulate_world(seed=841)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_planted_edge():
    _, truth = data.simulate_world(beta=0.0, seed=841)
    assert not truth.has_edge
    _, truth2 = data.simulate_world(beta=0.005, seed=841)
    assert truth2.has_edge


def test_predictor_is_persistent():
    """The predictor is a highly persistent AR(1) — the regime where the overlap trap bites hardest."""
    df, _ = data.simulate_world(n_months=4000, rho=0.95, seed=841)
    x = df["x"].to_numpy() - df["x"].to_numpy().mean()
    ac1 = float(np.sum(x[:-1] * x[1:]) / np.sum(x * x))
    assert ac1 > 0.85  # near the planted rho = 0.95


def test_null_returns_have_no_mean_predictability():
    """Under beta = 0 the one-period return is (near-)unpredictable from x: the one-period slope's
    contribution to R² is tiny, so any long-horizon R² is an overlap artefact, not real signal."""
    df, _ = data.simulate_world(n_months=6000, beta=0.0, rho=0.95, seed=841)
    x = df["x"].to_numpy()[:-1]
    r = df["r"].to_numpy()[1:]
    xd = x - x.mean()
    b = float(xd @ (r - r.mean()) / (xd @ xd))
    r2_1period = b**2 * (xd @ xd) / float((r - r.mean()) @ (r - r.mean()))
    assert r2_1period < 0.01  # essentially no one-period predictability


def test_planted_edge_lifts_one_period_predictability():
    df, _ = data.simulate_world(n_months=6000, beta=0.005, rho=0.95, seed=841)
    x = df["x"].to_numpy()[:-1]
    r = df["r"].to_numpy()[1:]
    xd = x - x.mean()
    b = float(xd @ (r - r.mean()) / (xd @ xd))
    assert b > 0.002  # the planted positive slope shows up


def test_stambaugh_feedback_sign():
    """The return/predictor innovation correlation (delta < 0) is present — the Stambaugh feedback."""
    df, _ = data.simulate_world(n_months=6000, beta=0.0, rho=0.95, delta=-0.9, seed=841)
    x = df["x"].to_numpy()
    r = df["r"].to_numpy()
    du = x[1:] - 0.95 * x[:-1]         # predictor innovation
    eps = r[1:] - r[1:].mean()          # (beta=0) return innovation ~ eps
    corr = np.corrcoef(du, eps)[0, 1]
    assert corr < -0.5                  # strongly negative, as planted
