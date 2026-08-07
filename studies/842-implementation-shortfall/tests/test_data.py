"""The synthetic tape is deterministic, offline, and carries the planted structure:
a genuine gross edge that loads on the LAGGED signal (no look-ahead), a signal whose
turnover is set by the persistence knob (independent of the edge), and a clean null."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cost_gap import data  # noqa: E402


def test_panel_deterministic(edge_world):
    rets, sig, _ = edge_world
    r2, s2, _ = data.synthetic_panel(edge=0.0005, persistence=0.96, n_assets=30,
                                     n_days=2520, seed=842)
    assert np.allclose(rets.to_numpy(), r2.to_numpy())
    assert np.allclose(sig.to_numpy(), s2.to_numpy())
    assert data.fingerprint(rets) == data.fingerprint(r2)


def test_null_has_no_planted_edge():
    _, _, t0 = data.synthetic_panel(edge=0.0, seed=842)
    _, _, t1 = data.synthetic_panel(edge=0.0005, seed=842)
    assert not t0.has_edge
    assert t1.has_edge


def test_return_loads_on_lagged_signal(edge_world):
    """The next-day return correlates with YESTERDAY's signal (the point-in-time edge):
    a clearly positive lag-1 signal->return correlation, while the null has none."""
    rets, sig, _ = edge_world
    R = rets.to_numpy()
    S = sig.to_numpy()
    lag_corr = np.corrcoef(S[:-1].ravel(), R[1:].ravel())[0, 1]
    assert lag_corr > 0.02
    null_rets, null_sig, _ = data.synthetic_panel(edge=0.0, persistence=0.96,
                                                  n_assets=30, n_days=2520, seed=842)
    Rn, Sn = null_rets.to_numpy(), null_sig.to_numpy()
    null_lag_corr = np.corrcoef(Sn[:-1].ravel(), Rn[1:].ravel())[0, 1]
    assert abs(null_lag_corr) < lag_corr


def test_signal_is_unit_variance_across_persistence():
    """The latent signal is standardised to ~unit variance for every phi, so the gross edge
    is a function of `edge` alone and NOT of the turnover that phi controls."""
    for phi in (0.995, 0.9, 0.3):
        _, sig, _ = data.synthetic_panel(edge=0.0005, persistence=phi, n_days=3000, seed=842)
        assert 0.7 < sig.to_numpy().std() < 1.3


def test_null_gross_mean_is_tiny(null_world):
    """On the null the cross-sectional return has no relation to the signal, so the raw
    panel's mean is ~0 (nothing for a sort to harvest)."""
    rets, _, _ = null_world
    assert abs(rets.to_numpy().mean()) < 1e-4
