"""The synthetic factor zoo is deterministic, offline, and behaves as advertised: the
null is pure noise (per-factor t-stats ~ N(0,1), so ~4.55% clear t>2 and ~0.27% clear
t>3), and the positive control plants a known, recoverable true subset."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tstat_threshold import data  # noqa: E402
from tstat_threshold import strategy as st  # noqa: E402


def test_zoo_deterministic():
    a, ma, _ = data.synthetic_zoo(seed=839)
    b, mb, _ = data.synthetic_zoo(seed=839)
    assert np.allclose(a, b)
    assert np.array_equal(ma, mb)
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_edge_mixture_does():
    _, m0, t0 = data.synthetic_zoo(n_true=0, seed=839)
    assert not t0.has_edge
    assert m0.sum() == 0
    _, m1, t1 = data.synthetic_zoo(n_true=50, seed=839)
    assert t1.has_edge
    assert int(m1.sum()) == 50
    assert abs(t1.frac_true - 0.05) < 1e-9


def test_shape_and_labels():
    R, is_true, truth = data.synthetic_zoo(n_factors=300, n_periods=180, n_true=20, seed=1)
    assert R.shape == (180, 300)
    assert is_true[:20].all() and not is_true[20:].any()
    assert truth.n_factors == 300 and truth.n_true == 20 and truth.n_periods == 180


def test_null_tstats_are_standard_normal():
    """Pure-noise per-factor t-stats behave like N(0,1): mean ~0, sd ~1 (averaged over
    seeds so a single lucky RNG can't drive the assertion)."""
    means, sds = [], []
    for s in range(839, 869):
        R, _, _ = data.synthetic_zoo(n_factors=1000, n_periods=240, n_true=0, seed=s)
        t = st.factor_tstats(R)
        means.append(t.mean()); sds.append(t.std(ddof=1))
    assert abs(np.mean(means)) < 0.05
    assert 0.93 < np.mean(sds) < 1.07


def test_null_clearing_fractions_match_theory():
    """~4.55% of noise factors clear t>2 and ~0.27% clear t>3 (20-seed average)."""
    sn = st.seed_robust_null(data, n_factors=1000, n_periods=240, n_seeds=20)
    assert abs(sn["mean_frac_gt2"] - st.prob_exceed(2.0)) < 0.01
    assert abs(sn["mean_frac_gt3"] - st.prob_exceed(3.0)) < 0.004
    assert sn["ratio_gt2_over_gt3"] > 8.0     # the lax bar admits >8x more noise


def test_planted_true_factors_are_strong():
    """A planted true factor (expected |t| = 4) posts a large single-test t on average."""
    R, is_true, _ = data.synthetic_zoo(n_true=50, expected_t=4.0, n_periods=240, seed=839)
    t = st.factor_tstats(R)
    assert np.mean(np.abs(t[is_true])) > 3.0   # the real ones really are significant
    assert np.mean(np.abs(t[~is_true])) < 1.5  # the noise ones are not


def test_fingerprint_changes_with_content():
    a, _, _ = data.synthetic_zoo(seed=839)
    b, _, _ = data.synthetic_zoo(seed=840)
    assert data.fingerprint(a) != data.fingerprint(b)
