"""The null-world generators are well-formed, deterministic, mean-zero, and carry exactly the
serial correlation their closed forms predict."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hac_necessity import data as d, strategy as st  # noqa: E402


# ---- shape & determinism ----------------------------------------------------
def test_overlap_shape_and_finite(overlap_null):
    assert overlap_null.shape == (400, 2000)
    assert np.isfinite(overlap_null).all()


def test_overlap_deterministic():
    a = d.overlap_matrix(20, 500, window=21, seed=5)
    b = d.overlap_matrix(20, 500, window=21, seed=5)
    c = d.overlap_matrix(20, 500, window=21, seed=6)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_ar1_deterministic():
    a = d.ar1_matrix(20, 500, rho=0.5, seed=5)
    b = d.ar1_matrix(20, 500, rho=0.5, seed=5)
    assert np.allclose(a, b)


def test_series_wrappers_match_matrix():
    s = d.overlap_returns(300, window=21, seed=9)
    m = d.overlap_matrix(1, 300, window=21, seed=9)[0]
    assert np.allclose(s.to_numpy(), m)
    # decorative index does not overflow ns-timestamps
    assert s.index.is_monotonic_increasing and s.index[0].year >= 1900


# ---- the null really is mean-zero ------------------------------------------
def test_overlap_null_is_mean_zero():
    """Averaged over many paths the per-path mean centres on zero (it is a null)."""
    X = d.overlap_matrix(2000, 3000, window=21, seed=1, mean=0.0)
    grand_mean = X.mean()
    assert abs(grand_mean) < 5e-5


def test_mean_knob_shifts_the_level():
    a = d.overlap_matrix(50, 2000, window=21, seed=2, mean=0.0).mean()
    b = d.overlap_matrix(50, 2000, window=21, seed=2, mean=6e-4).mean()
    assert b - a > 4e-4  # the planted mean shows up


# ---- the injected autocorrelation matches its closed form ------------------
def test_overlap_autocorrelation_matches_theory():
    """A window-K overlap is MA(K-1) with lag-l autocorr (K-l)/K."""
    x = d.overlap_returns(60000, window=21, seed=3).to_numpy()
    ac = st.autocorr(x, lags=25)
    assert abs(ac[0] - 20 / 21) < 0.02          # lag 1
    assert abs(ac[4] - 16 / 21) < 0.03          # lag 5
    assert abs(ac[20]) < 0.03                    # lag 21 -> ~0 (MA(20) cuts off)


def test_ar1_autocorrelation_matches_theory():
    """AR(1) lag-l autocorrelation is rho**l."""
    x = d.ar1_returns(80000, rho=0.6, seed=4).to_numpy()
    ac = st.autocorr(x, lags=3)
    assert abs(ac[0] - 0.6) < 0.02
    assert abs(ac[1] - 0.36) < 0.02


def test_theoretical_inflation_forms():
    assert abs(d.theoretical_inflation_overlap(21) - np.sqrt(21)) < 1e-9
    assert abs(d.theoretical_inflation_ar1(0.6) - np.sqrt(1.6 / 0.4)) < 1e-9
    assert d.theoretical_inflation_overlap(1) == 1.0
    assert d.theoretical_inflation_ar1(0.0) == 1.0


# ---- fingerprints -----------------------------------------------------------
def test_fingerprint_stable_and_sensitive(overlap_null):
    assert d.fingerprint(overlap_null) == d.fingerprint(overlap_null)
    other = d.overlap_matrix(400, 2000, window=21, seed=999)
    assert d.fingerprint(overlap_null) != d.fingerprint(other)


def test_config_fingerprint_is_stable():
    assert d.config_fingerprint() == d.config_fingerprint()
    assert len(d.config_fingerprint()) == 12
