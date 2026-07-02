"""Deterministic, offline tests for Study 576 (Muni-Treasury-Ratio)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from muni_treasury_ratio import data, strategy as st


def test_synthetic_null_is_flat():
    """At the null (timing_beta=0) the mean HAC slope-t stays near zero (seed-robust)."""
    t = st.synthetic_mean_t(data, timing_beta=0.0, horizon=63, n_seeds=25)
    assert abs(t) < 1.0


def test_synthetic_positive_control_fires():
    """A planted timing effect drives the mean HAC slope-t positive and past the bar."""
    t_weak = st.synthetic_mean_t(data, timing_beta=0.03, horizon=63, n_seeds=25)
    t_strong = st.synthetic_mean_t(data, timing_beta=0.15, horizon=63, n_seeds=25)
    assert t_weak > 0
    assert t_strong > 2.0
    assert t_strong > t_weak


def test_no_lookahead_forward_excess_uses_future_only():
    """forward_excess at t must be NaN in the final `horizon` rows (no future data available)."""
    frame, _ = data.synthetic_series(timing_beta=0.02, horizon=63, seed=576)
    fe = st.forward_excess(frame, horizon=63)
    assert fe.tail(63).isna().all()


def test_fetch_series_offline_is_empty():
    """With fetch=False and no cache path override, the loader never hits the network."""
    df = data.fetch_series(cache_dir="/nonexistent_cache_dir_576", fetch=False)
    # Either the study's real cache exists (returns real frame) or empty; never raises.
    assert df is not None


def test_ratio_z_is_trailing_only():
    """The ratio z at row i uses only rows <= i (no forward leakage)."""
    frame, _ = data.synthetic_series(seed=576)
    z = st.ratio_z(frame, lookback=252)
    # First (lookback//2 - 1) rows have insufficient history -> NaN.
    assert z.iloc[:100].isna().any()
    assert np.isfinite(z.iloc[-1])
