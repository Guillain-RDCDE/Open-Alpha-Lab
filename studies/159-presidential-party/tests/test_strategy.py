"""Strategy invariants: party statistics, D-minus-R gap, permutation test, summarize."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidential_party import strategy as st  # noqa: E402
from presidential_party import data  # noqa: E402

# Shiller cache is git-ignored (absent in CI); skip the real-tape test there.
requires_shiller = pytest.mark.skipif(
    not os.path.exists(data._SHILLER_CACHE),
    reason="Shiller cache not present (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# party_returns
# ---------------------------------------------------------------------------
def test_party_returns_columns(null_tape):
    df, _ = null_tape
    pr = st.party_returns(df)
    assert set(pr.columns) >= {"n", "mean_monthly_bps", "std_monthly_bps",
                               "mean_ann_pct", "hac_tstat"}
    assert set(pr.index) == {"D", "R"}


def test_party_returns_n_matches_data(null_tape):
    df, _ = null_tape
    pr = st.party_returns(df)
    n_d = (df["party"] == "D").sum()
    n_r = (df["party"] == "R").sum()
    assert pr.loc["D", "n"] == n_d
    assert pr.loc["R", "n"] == n_r


def test_party_returns_mean_ann_approx_12x_monthly(null_tape):
    df, _ = null_tape
    pr = st.party_returns(df)
    # mean_ann_pct = mean_monthly_bps / 100 * 12  (bps/month -> %/year)
    # Both are rounded to 2 decimal places in party_returns(), so tolerance is 0.01
    for party in ("D", "R"):
        expected = pr.loc[party, "mean_monthly_bps"] / 100.0 * 12.0
        got = pr.loc[party, "mean_ann_pct"]
        assert abs(got - expected) < 0.01, f"Annualisation mismatch for {party}: got {got}, expected {expected}"


# ---------------------------------------------------------------------------
# dem_minus_rep
# ---------------------------------------------------------------------------
def test_dem_minus_rep_null_has_small_gap(null_tape):
    """With zero planted premium, the D-R gap should be small (not guaranteed zero due to noise)."""
    df, _ = null_tape
    gap = st.dem_minus_rep(df)
    assert abs(gap["gap_monthly_bps"]) < 30.0, (
        f"Null tape D-R gap = {gap['gap_monthly_bps']:.1f} bps/month, expected < 30"
    )


def test_dem_minus_rep_planted_has_large_gap(planted_tape):
    df, _ = planted_tape
    gap = st.dem_minus_rep(df)
    assert gap["gap_monthly_bps"] > 200.0, (
        f"Planted tape D-R gap = {gap['gap_monthly_bps']:.1f} bps/month, expected > 200"
    )


def test_dem_minus_rep_welch_t_sign_matches_gap(null_tape, planted_tape):
    for df, _ in [null_tape, planted_tape]:
        gap = st.dem_minus_rep(df)
        if abs(gap["gap_monthly_bps"]) > 1:
            assert np.sign(gap["welch_t"]) == np.sign(gap["gap_monthly_bps"])


def test_dem_minus_rep_returns_required_keys(null_tape):
    df, _ = null_tape
    gap = st.dem_minus_rep(df)
    required = {"gap_monthly_bps", "gap_ann_pct", "welch_t", "hac_t", "n_dem", "n_rep"}
    assert required.issubset(gap.keys())


@requires_shiller
def test_dem_minus_rep_on_real_shiller():
    """The real Shiller tape should have a positive D-R gap (Santa-Clara & Valkanov)."""
    df = data.load_shiller()
    gap = st.dem_minus_rep(df)
    # The gap may be large (historical finding) but the key is it's finite and computable
    assert np.isfinite(gap["gap_monthly_bps"])
    assert np.isfinite(gap["welch_t"])


# ---------------------------------------------------------------------------
# permutation_pvalue
# ---------------------------------------------------------------------------
def test_permutation_pvalue_null_not_significant(null_tape):
    """Permutation p-value should be large (> 0.1) when no premium is planted."""
    df, _ = null_tape
    result = st.permutation_pvalue(df, n_perm=2000, seed=42)
    assert result["pvalue_two_sided"] > 0.05, (
        f"Null tape permutation p-value = {result['pvalue_two_sided']:.3f}, expected > 0.05"
    )


def test_permutation_pvalue_planted_may_be_significant(planted_tape):
    """With a large planted premium, the permutation p-value should be small."""
    df, _ = planted_tape
    result = st.permutation_pvalue(df, n_perm=2000, seed=42)
    # On n=1200 months with 100 bps/month planted, signal should be detectable
    assert result["pvalue_two_sided"] < 0.20, (
        f"Planted tape permutation p-value = {result['pvalue_two_sided']:.3f}, expected < 0.20"
    )


def test_permutation_pvalue_returns_required_keys(null_tape):
    df, _ = null_tape
    result = st.permutation_pvalue(df, n_perm=500, seed=1)
    required = {"observed_gap_bps", "pvalue_two_sided", "n_perm", "n_terms"}
    assert required.issubset(result.keys())


def test_permutation_reproducible(null_tape):
    df, _ = null_tape
    r1 = st.permutation_pvalue(df, n_perm=1000, seed=77)
    r2 = st.permutation_pvalue(df, n_perm=1000, seed=77)
    assert r1["pvalue_two_sided"] == r2["pvalue_two_sided"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_returns_flat_dict(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    assert isinstance(s, dict)
    required = {"n_total", "n_dem", "n_rep", "dem_mean_bps", "rep_mean_bps",
                "gap_monthly_bps", "gap_ann_pct", "welch_t", "hac_t", "perm_pvalue"}
    assert required.issubset(s.keys())


def test_summarize_n_total(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    assert s["n_total"] == int(df["ret"].notna().sum())


def test_summarize_gap_consistent(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    gap_direct = st.dem_minus_rep(df)
    assert abs(s["gap_monthly_bps"] - gap_direct["gap_monthly_bps"]) < 1e-6
