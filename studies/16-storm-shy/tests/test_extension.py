"""The beat-7 complement: the no-borrowing test. Capping leverage at 1.0 isolates the *de-risk* slice
(cutting exposure in storms — needs no borrowing) from the *leverage* slice (gearing up in calm —
the contested Cederburg part). On a clustered tape a real share of the Sharpe gain — and essentially
all of the drawdown reduction — survives with no borrowing; on the flat null nothing survives at any
cap; and the decomposition is an exact split."""

import numpy as np

from storm_shy import extension


def test_decomposition_is_an_exact_split(regime_returns):
    """gain_full = gain_derisk + gain_leverage, to machine precision (it's two compares differenced)."""
    g = extension.gain_decomposition(regime_returns)
    assert np.isclose(g["gain_full"], g["gain_derisk"] + g["gain_leverage"], atol=1e-12)


def test_derisk_slice_survives_no_borrowing(regime_returns):
    """A real chunk of the Sharpe gain is there with leverage capped at 1.0 — and it needs no
    borrowing (average leverage <= 1), because cutting risk in storms doesn't require gearing up."""
    g = extension.gain_decomposition(regime_returns)
    assert g["gain_derisk"] > 0.15
    assert g["derisk_avg_leverage"] <= 1.0 + 1e-9
    assert 0.3 < g["share_derisk"] < 1.2          # most of the gain, not levered into existence


def test_drawdown_reduction_needs_no_leverage(regime_returns):
    """The visceral benefit — shallower drawdowns — is a de-risk effect: it survives the 1.0 cap."""
    g = extension.gain_decomposition(regime_returns)
    assert g["derisk_maxdd"] > g["buyhold_maxdd"]      # shallower (less negative) with no borrowing


def test_leverage_slice_is_real_on_synthetic(regime_returns):
    """On the idealized constant-drift tape, gearing up in calm genuinely adds a (smaller) slice —
    so gain_full strictly exceeds the de-risk-only gain."""
    g = extension.gain_decomposition(regime_returns)
    assert g["gain_leverage"] > 0.02


def test_nothing_survives_on_the_flat_null(flat_returns):
    """No regime -> no gain at any cap; the whole decomposition collapses to ~0."""
    g = extension.gain_decomposition(flat_returns)
    assert abs(g["gain_full"]) < 0.1
    assert abs(g["gain_derisk"]) < 0.1


def test_cap_sweep_is_monotone(regime_returns):
    """Relaxing the leverage cap can only add exposure, so both the Sharpe gain and the average
    leverage are non-decreasing down the table."""
    sw = extension.leverage_cap_sweep(regime_returns)
    assert (np.diff(sw["sharpe_gain"].to_numpy()) >= -1e-9).all()
    assert (np.diff(sw["avg_leverage"].to_numpy()) >= -1e-9).all()
    assert sw.loc[1.0, "avg_leverage"] <= 1.0 + 1e-9
