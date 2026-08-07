"""The engine and the study's spine: (1) overlapping long-horizon returns inflate the NAIVE OLS t and
R² under a true null, monotonically in the horizon h; (2) the Hodrick 1B standard error restores
honest size while Newey-West only partly does; (3) the corrections still DETECT a genuinely planted
edge (power, not just size). Plus the mechanical primitives: the overlap builder, the OLS slope, and
the Hodrick numerator identity."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overlapping_returns import data  # noqa: E402
from overlapping_returns import strategy as st  # noqa: E402


# ---- the overlap builder ---------------------------------------------------
def test_overlapping_returns_values_and_length():
    r = np.arange(1.0, 11.0)  # 1..10
    y = st.overlapping_returns(r, 3)
    assert len(y) == len(r) - 3
    # y[0] = r[1]+r[2]+r[3] = 2+3+4 = 9 ; y[1] = 3+4+5 = 12
    assert np.isclose(y[0], 9.0)
    assert np.isclose(y[1], 12.0)


def test_overlapping_returns_h1_is_the_next_return():
    r = np.array([0.1, 0.2, -0.3, 0.4])
    y = st.overlapping_returns(r, 1)
    assert np.allclose(y, r[1:])


# ---- OLS slope + naive t ---------------------------------------------------
def test_ols_recovers_planted_line():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 2.0 + 3.0 * x + 0.01 * rng.standard_normal(500)
    out = st.ols_slope_t(x, y)
    assert abs(out["slope"] - 3.0) < 0.01
    assert out["t_naive"] > 50           # a real, tight relationship
    assert out["r2"] > 0.99


# ---- Hodrick numerator identity (the estimator's algebraic core) -----------
def test_hodrick_numerator_identity(null_world):
    df, _ = null_world
    x = df["x"].to_numpy(); r = df["r"].to_numpy()
    h = 12
    N = len(r) - h
    xd = x[:N] - x[:N].mean()
    y = st.overlapping_returns(r, h)
    num_overlap = float(xd @ (y - y.mean()))
    # reconstruct the summed-regressor numerator sum_s r_s * XS_s
    xd_pad = np.zeros(len(r)); xd_pad[:N] = xd
    cs = np.concatenate([[0.0], np.cumsum(xd_pad)])
    s = np.arange(len(r))
    XS = cs[np.clip(s, 0, len(r))] - cs[np.clip(s - h, 0, len(r))]
    num_summed = float(r @ XS)
    assert abs(num_overlap - num_summed) < 1e-8


# ---- no overlap (h=1): all three standard errors agree ---------------------
def test_h1_no_overlap_all_agree(null_world):
    df, _ = null_world
    o = st.predictive_regression(df["x"].to_numpy(), df["r"].to_numpy(), h=1)
    # with no overlap the three t-stats are essentially the same
    assert abs(o["t_naive"] - o["t_nw"]) < 0.3
    assert abs(o["t_naive"] - o["t_hodrick"]) < 0.3


# ---- one world: the naive t spuriously "discovers" what Hodrick rejects ----
def test_single_null_world_naive_inflated(null_world):
    df, _ = null_world
    o = st.predictive_regression(df["x"].to_numpy(), df["r"].to_numpy(), h=12)
    assert abs(o["t_naive"]) > 1.96            # naive: spuriously "significant"
    assert abs(o["t_hodrick"]) < 1.96          # Hodrick: correctly NOT significant
    assert abs(o["t_naive"]) > abs(o["t_hodrick"]) + 1.5
    assert abs(o["t_nw"]) < abs(o["t_naive"])  # NW deflates the naive t


# ---- THE HEADLINE: naive over-sized, Hodrick well-sized, under the null -----
def test_naive_oversized_hodrick_sized_h12():
    e = st.size_experiment(data, h=12, beta=0.0, rho=0.95, n_sims=250, base_seed=841)
    assert e["reject_naive"] > 0.35           # a 5% test rejecting the null a THIRD+ of the time
    assert e["reject_hodrick"] < 0.12          # Hodrick ~ nominal 5%
    assert e["reject_hodrick"] < e["reject_naive"]
    assert e["reject_nw"] < e["reject_naive"]  # NW helps


def test_size_distortion_grows_with_horizon():
    e1 = st.size_experiment(data, h=1, beta=0.0, n_sims=250, base_seed=841)
    e24 = st.size_experiment(data, h=24, beta=0.0, n_sims=250, base_seed=841)
    assert e1["reject_naive"] < 0.12                          # honest at h=1 (no overlap)
    assert e24["reject_naive"] > e1["reject_naive"] + 0.25    # much worse at h=24
    assert e24["reject_hodrick"] < 0.12                       # Hodrick stays sized
    assert e24["mean_r2"] > e1["mean_r2"]                     # naive R² inflates with h too


# ---- the control: the corrections DETECT a real edge (power) ---------------
def test_corrections_have_power_on_planted_edge():
    e = st.size_experiment(data, h=6, beta=0.005, rho=0.95, n_sims=200, base_seed=841)
    assert e["reject_hodrick"] > 0.6           # Hodrick still finds a genuine edge
    assert e["reject_nw"] > 0.6                 # so does Newey-West


def test_corrections_sized_under_null_powered_under_edge():
    null = st.size_experiment(data, h=6, beta=0.0, n_sims=200, base_seed=841)
    edge = st.size_experiment(data, h=6, beta=0.005, n_sims=200, base_seed=841)
    # Hodrick: near-nominal size under the null, high power under the edge
    assert null["reject_hodrick"] < 0.12
    assert edge["reject_hodrick"] > null["reject_hodrick"] + 0.4
