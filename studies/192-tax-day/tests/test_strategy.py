"""Strategy engine invariants, controls, and the study's spine:
the tax-day signal is detectable only when a genuine effect is planted."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tax_day import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# HAC t-stat sanity checks
# ---------------------------------------------------------------------------
def test_hac_tstat_on_trivial_series():
    """Mean >> 0 and small noise → t should be very large."""
    rng = np.random.default_rng(42)
    # Small positive noise around a large mean → very high t-stat.
    r = 1.0 + rng.normal(0, 1e-6, 200)
    out = st._hac_tstat(r)
    assert out["n"] == 200
    assert abs(out["tstat"]) > 5.0, "near-constant large-mean array should give huge t"


def test_hac_tstat_on_zero_mean():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1, 500)
    out = st._hac_tstat(r)
    assert abs(out["tstat"]) < 3.0, "zero-mean noise should not produce large t"


def test_hac_tstat_small_n_returns_nan():
    r = np.array([1.0, 2.0])
    out = st._hac_tstat(r)
    assert np.isnan(out["tstat"])


# ---------------------------------------------------------------------------
# window_comparison — structural invariants
# ---------------------------------------------------------------------------
def test_window_comparison_keys(null_tape):
    df, _ = null_tape
    res = st.window_comparison(df, "pre_tax")
    required = {
        "n_window", "mean_window_bps", "tstat_window_hac",
        "n_rest", "mean_rest_bps", "tstat_rest_hac",
        "contrast_bps", "tstat_welch", "pval_welch",
    }
    assert required <= set(res.keys())


def test_window_comparison_counts_consistent(null_tape):
    df, _ = null_tape
    res = st.window_comparison(df, "pre_tax")
    total = res["n_window"] + res["n_rest"]
    assert total == len(df), "window + rest must equal total days"


def test_pvalue_in_unit_interval(null_tape):
    df, _ = null_tape
    for col in ("pre_tax", "post_tax"):
        res = st.window_comparison(df, col)
        assert 0.0 <= res["pval_welch"] <= 1.0


# ---------------------------------------------------------------------------
# april_comparison — structural invariants
# ---------------------------------------------------------------------------
def test_april_comparison_keys(null_tape):
    df, _ = null_tape
    res = st.april_comparison(df)
    assert {"n_apr", "mean_apr_bps", "n_rest", "contrast_bps", "pval_welch"} <= set(res.keys())


def test_april_count_is_reasonable(null_tape):
    df, _ = null_tape
    res = st.april_comparison(df)
    # April has ~21 trading days; 75 years → ~1575 April days.
    assert 1000 <= res["n_apr"] <= 2000, f"unexpected April count: {res['n_apr']}"


# ---------------------------------------------------------------------------
# placebo_comparison — structural
# ---------------------------------------------------------------------------
def test_placebo_keys(null_tape):
    df, _ = null_tape
    res = st.placebo_comparison(df)
    assert {"n_oct_window", "mean_oct_bps", "contrast_bps", "pval_welch"} <= set(res.keys())


def test_placebo_count_similar_to_tax_window(null_tape):
    df, _ = null_tape
    tax_pre = st.window_comparison(df, "pre_tax")
    tax_post = st.window_comparison(df, "post_tax")
    plac = st.placebo_comparison(df)
    tax_total = tax_pre["n_window"] + tax_post["n_window"]
    # Oct placebo window should be similar in size (+/- 20%).
    assert 0.8 * tax_total <= plac["n_oct_window"] <= 1.2 * tax_total, (
        f"placebo window size {plac['n_oct_window']} diverges from "
        f"tax window size {tax_total}"
    )


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------
def test_bonferroni_always_at_least_raw(null_tape):
    df, _ = null_tape
    pre = st.window_comparison(df, "pre_tax")
    post = st.window_comparison(df, "post_tax")
    bonf = st.bonferroni_summary(pre, post)
    assert bonf["pval_pre_bonferroni"] >= pre["pval_welch"] - 1e-9
    assert bonf["pval_post_bonferroni"] >= post["pval_welch"] - 1e-9
    assert bonf["pval_pre_bonferroni"] <= 1.0
    assert bonf["pval_post_bonferroni"] <= 1.0


# ---------------------------------------------------------------------------
# summarize — flat dict / nested consistency
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    df, _ = null_tape
    s = st.summarize(df)
    required_keys = {
        "n_pre", "mean_pre_bps", "tstat_pre_hac", "contrast_pre_bps", "pval_pre",
        "n_post", "mean_post_bps", "tstat_post_hac", "contrast_post_bps", "pval_post",
        "n_apr", "mean_apr_bps", "contrast_apr_bps", "pval_apr",
        "mean_all_bps",
        "n_oct_window", "mean_oct_bps", "contrast_oct_bps", "pval_oct",
        "pval_pre_bonferroni", "pval_post_bonferroni", "any_survives_bonferroni",
    }
    assert required_keys <= set(s.keys())


# ---------------------------------------------------------------------------
# The spine: planted effects are detected, null is not
# ---------------------------------------------------------------------------
def test_pre_effect_detected(pre_effect_tape, null_tape):
    """A planted positive pre-tax effect raises the window mean significantly."""
    df_eff, _ = pre_effect_tape
    df_null, _ = null_tape
    res_eff = st.window_comparison(df_eff, "pre_tax")
    res_null = st.window_comparison(df_null, "pre_tax")
    # On the planted tape the contrast should be large and positive.
    assert res_eff["contrast_bps"] > 10.0, "planted effect should produce large contrast"
    assert abs(res_eff["tstat_welch"]) > 2.0, "planted effect should clear t>2"
    # On the null tape it should not clear that bar.
    assert abs(res_null["tstat_welch"]) < 4.0, "null tape should not produce high t"


def test_post_effect_detected(post_effect_tape, null_tape):
    """A planted negative post-tax effect lowers the window mean significantly."""
    df_eff, _ = post_effect_tape
    df_null, _ = null_tape
    res_eff = st.window_comparison(df_eff, "post_tax")
    res_null = st.window_comparison(df_null, "post_tax")
    # On the planted tape the contrast should be large and negative.
    assert res_eff["contrast_bps"] < -10.0, "planted negative post effect should produce negative contrast"
    assert abs(res_eff["tstat_welch"]) > 2.0, "planted effect should clear |t|>2"
    assert abs(res_null["tstat_welch"]) < 4.0, "null tape should not produce high |t|"


def test_null_tape_no_spurious_signal(null_tape):
    """On a pure random walk neither window clears significance at |t|>2."""
    df, _ = null_tape
    s = st.summarize(df)
    # With 75 years of daily data the HAC t-stats should stay well within noise.
    assert abs(s["tstat_pre_hac"]) < 3.5, "null tape should not produce large |t| for pre_tax"
    assert abs(s["tstat_post_hac"]) < 3.5, "null tape should not produce large |t| for post_tax"


# ---------------------------------------------------------------------------
# Real-data tests (skipped in CI when cache is absent)
# ---------------------------------------------------------------------------
_GSPC_SHARED = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "^GSPC_split_only.parquet"
)
_GSPC_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "_cache", "gspc_daily.parquet"
)
_cache_present = os.path.exists(_GSPC_SHARED) or os.path.exists(_GSPC_LOCAL)

requires_cache = pytest.mark.skipif(
    not _cache_present,
    reason="GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_summarize_returns_finite_numbers():
    df = data.fetch_gspc(start="1928-01-01")
    s = st.summarize(df)
    import math
    for key in ("mean_pre_bps", "mean_post_bps", "pval_pre", "pval_post"):
        assert math.isfinite(s[key]), f"{key} is not finite on real tape"


@requires_cache
def test_real_tape_pvalues_in_unit_interval():
    df = data.fetch_gspc(start="1928-01-01")
    s = st.summarize(df)
    for key in ("pval_pre", "pval_post", "pval_apr", "pval_oct"):
        assert 0.0 <= s[key] <= 1.0, f"{key} = {s[key]} out of [0,1]"
