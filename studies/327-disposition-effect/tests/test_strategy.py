"""The cross-sectional engine's invariants and the study's two spines:
(1) the overhang hedge clears the bar ONLY when the disposition premium is really planted;
(2) a momentum-only world makes the *raw* overhang hedge fire spuriously, and orthogonalising
    to momentum kills it — i.e. 'disposition' must be shown to be more than momentum."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from disposition_effect import strategy as st  # noqa: E402


# ---- engine invariants ----------------------------------------------------
def test_quintile_hedge_columns(planted):
    over, _, fwd, _ = planted
    h = st.quintile_hedge(over, fwd)
    for c in ("Q1", "Q5", "hedge", "market", "n_firms"):
        assert c in h.columns
    assert (h["n_firms"] > 0).all()


def test_hedge_is_top_minus_bottom(planted):
    over, _, fwd, _ = planted
    h = st.quintile_hedge(over, fwd)
    assert np.allclose(h["hedge"], h["Q5"] - h["Q1"])


def test_long_low_flips_sign(planted):
    over, _, fwd, _ = planted
    hi = st.quintile_hedge(over, fwd, long_high=True)["hedge"]
    lo = st.quintile_hedge(over, fwd, long_high=False)["hedge"]
    assert np.allclose(hi, -lo)


def test_costs_lower_mean_monotonically(planted):
    over, _, fwd, _ = planted
    h = st.quintile_hedge(over, fwd)["hedge"]
    means = [st.apply_costs(h, c).mean() for c in (0.0, 2.0, 5.0)]
    assert means[0] > means[1] > means[2]


def test_apply_costs_formula(planted):
    over, _, fwd, _ = planted
    h = st.quintile_hedge(over, fwd)["hedge"]
    net = st.apply_costs(h, cost_bps=5.0, turnover_per_leg=2.0)
    assert np.allclose(net, h - 2.0 * 2.0 * 5.0 * 1e-4)


def test_block_bootstrap_ci_brackets_mean(planted):
    over, _, fwd, _ = planted
    h = st.quintile_hedge(over, fwd)["hedge"]
    lo, hi = st.block_bootstrap_ci(h, n_boot=500)
    assert lo < h.mean() < hi


# ---- spine #1: the hedge fires only with a planted premium -----------------
def test_hedge_real_only_when_premium_planted(planted, null):
    over_p, _, fwd_p, _ = planted
    over_0, _, fwd_0, _ = null
    t_p = st.summary(st.quintile_hedge(over_p, fwd_p)["hedge"])["tstat"]
    t_0 = st.summary(st.quintile_hedge(over_0, fwd_0)["hedge"])["tstat"]
    assert t_p > 2.0          # planted -> clears the inference bar
    assert abs(t_0) < 2.0     # null -> indistinguishable from zero


def test_information_coefficient_positive_when_planted(planted, null):
    over_p, _, fwd_p, _ = planted
    over_0, _, fwd_0, _ = null
    ic_p = st.information_coefficient(over_p, fwd_p)
    ic_0 = st.information_coefficient(over_0, fwd_0)
    assert st.hac_tstat(ic_p) > 2.0
    assert abs(st.hac_tstat(ic_0)) < 2.5


# ---- spine #2: disposition must be more than momentum ----------------------
def test_orthogonalisation_kills_the_momentum_costume(momentum_only):
    """In a world where ONLY momentum is priced, the raw overhang hedge fires (overhang
    is correlated with momentum). Orthogonalising overhang to momentum collapses it —
    so a real disposition premium must survive that residual sort."""
    over, mom, fwd, truth = momentum_only
    assert not truth.has_overhang_edge
    raw_t = st.summary(st.quintile_hedge(over, fwd)["hedge"])["tstat"]
    resid = st.orthogonalise(over, mom)
    res_t = st.summary(st.quintile_hedge(resid, fwd)["hedge"])["tstat"]
    assert raw_t > 2.0                 # the costume: raw overhang looks 'real'
    assert res_t < raw_t * 0.6         # residual is materially weaker once mom is removed


def test_orthogonalise_residual_uncorrelated_with_control(planted):
    """The residual overhang should be ~uncorrelated with momentum cross-sectionally."""
    over, mom, _, _ = planted
    resid = st.orthogonalise(over, mom)
    common = resid.dropna(how="all").index.intersection(mom.index)
    cors = []
    for m in common:
        a = resid.loc[m]
        b = mom.loc[m]
        ok = a.notna() & b.notna()
        if ok.sum() > 5:
            cors.append(np.corrcoef(a[ok], b[ok])[0, 1])
    assert abs(np.nanmean(cors)) < 0.05


def test_hac_tstat_handles_short_series():
    assert np.isnan(st.hac_tstat(pd.Series([1.0, 2.0])))
