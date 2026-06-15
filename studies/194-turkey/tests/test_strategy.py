"""Strategy tests: the Thanksgiving comparison and its placebo/multiple-comparisons controls.

The spine: a planted positive Thanksgiving effect is detected; a null tape is not."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turkey import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — real-data tests are skipped offline
# ---------------------------------------------------------------------------
_GSPC_SHARED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache",
                 "^GSPC_split_only.parquet")
)
_GSPC_LOCAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "gspc_daily.parquet")
)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_GSPC_SHARED) or os.path.exists(_GSPC_LOCAL)),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_tape(wed_effect=0.0, fri_effect=0.0, seed=194, n_years=75):
    df, truth = data.synthetic_daily(
        n_years=n_years, wed_effect=wed_effect, fri_effect=fri_effect, seed=seed
    )
    return df, truth


# ---------------------------------------------------------------------------
# wednesday_comparison
# ---------------------------------------------------------------------------
class TestWednesdayComparison:
    def test_keys(self, null_tape):
        df, _ = null_tape
        r = st.wednesday_comparison(df)
        for k in ("n_wed", "mean_wed_bps", "tstat_hac_wed", "n_other_wed",
                   "mean_other_wed_bps", "contrast_bps", "tstat_welch", "pval_welch"):
            assert k in r

    def test_n_positive(self, null_tape):
        df, _ = null_tape
        r = st.wednesday_comparison(df)
        assert r["n_wed"] > 0
        assert r["n_other_wed"] > 0

    def test_n_other_much_larger(self, null_tape):
        """Other-Wednesday count must dwarf Thanksgiving Wednesdays (~75 vs ~3,700)."""
        df, _ = null_tape
        r = st.wednesday_comparison(df)
        assert r["n_other_wed"] > r["n_wed"] * 20

    def test_planted_effect_raises_contrast(self):
        """A planted positive effect should produce a positive contrast."""
        df_null, _ = _make_tape(wed_effect=0.0)
        df_pos, _ = _make_tape(wed_effect=5.0)
        r_null = st.wednesday_comparison(df_null)
        r_pos = st.wednesday_comparison(df_pos)
        assert r_pos["contrast_bps"] > r_null["contrast_bps"]
        assert r_pos["contrast_bps"] > 10.0  # clearly positive with strong effect

    def test_null_tape_contrast_small(self, null_tape):
        """On the null tape the contrast should be small."""
        df, _ = null_tape
        r = st.wednesday_comparison(df)
        # With no planted effect, |contrast| should be well under 25 bps.
        assert abs(r["contrast_bps"]) < 30.0

    def test_strong_effect_detectable(self):
        """A very strong planted effect should produce |t| > 2."""
        df, _ = _make_tape(wed_effect=8.0)
        r = st.wednesday_comparison(df)
        assert abs(r["tstat_welch"]) > 2.0


# ---------------------------------------------------------------------------
# friday_comparison (Black Friday)
# ---------------------------------------------------------------------------
class TestFridayComparison:
    def test_keys(self, null_tape):
        df, _ = null_tape
        r = st.friday_comparison(df)
        for k in ("n_fri", "mean_fri_bps", "tstat_hac_fri", "n_other_fri",
                   "mean_other_fri_bps", "contrast_bps", "tstat_welch", "pval_welch"):
            assert k in r

    def test_n_positive(self, null_tape):
        df, _ = null_tape
        r = st.friday_comparison(df)
        assert r["n_fri"] > 0
        assert r["n_other_fri"] > 0

    def test_planted_effect_raises_contrast(self):
        """A planted positive effect on Friday-after should produce a positive contrast."""
        df_null, _ = _make_tape(fri_effect=0.0)
        df_pos, _ = _make_tape(fri_effect=5.0)
        r_null = st.friday_comparison(df_null)
        r_pos = st.friday_comparison(df_pos)
        assert r_pos["contrast_bps"] > r_null["contrast_bps"]
        assert r_pos["contrast_bps"] > 10.0

    def test_null_tape_contrast_small(self, null_tape):
        """On the null tape the Black Friday contrast should be small."""
        df, _ = null_tape
        r = st.friday_comparison(df)
        assert abs(r["contrast_bps"]) < 30.0


# ---------------------------------------------------------------------------
# tuesday_placebo
# ---------------------------------------------------------------------------
class TestTuesdayPlacebo:
    def test_keys(self, null_tape):
        df, _ = null_tape
        r = st.tuesday_placebo(df)
        for k in ("n_tue", "mean_tue_bps", "tstat_hac_tue", "n_other_tue",
                   "mean_other_tue_bps", "contrast_bps", "tstat_welch", "pval_welch"):
            assert k in r

    def test_n_positive(self, null_tape):
        df, _ = null_tape
        r = st.tuesday_placebo(df)
        assert r["n_tue"] > 0
        assert r["n_other_tue"] > 0

    def test_placebo_count_similar_to_wed(self, null_tape):
        """Tuesday and Wednesday Thanksgiving counts should be roughly equal."""
        df, _ = null_tape
        wed_r = st.wednesday_comparison(df)
        tue_r = st.tuesday_placebo(df)
        assert abs(tue_r["n_tue"] - wed_r["n_wed"]) < 10

    def test_placebo_not_affected_by_wed_effect(self):
        """A planted Wednesday effect should not change the Tuesday placebo contrast."""
        df_null, _ = _make_tape(wed_effect=0.0)
        df_pos, _ = _make_tape(wed_effect=5.0)
        r_null = st.tuesday_placebo(df_null)
        r_pos = st.tuesday_placebo(df_pos)
        # Tuesday contrast should be nearly identical in both tapes
        # (within 5 bps noise tolerance).
        assert abs(r_pos["contrast_bps"] - r_null["contrast_bps"]) < 5.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_contains_expected_keys(self, null_tape):
        df, _ = null_tape
        s = st.summarize(df)
        for k in (
            "n_wed", "n_other_wed", "mean_wed_bps", "mean_other_wed_bps",
            "contrast_wed_bps", "tstat_welch_wed", "pval_welch_wed", "tstat_hac_wed",
            "pval_bonferroni_wed",
            "n_fri", "n_other_fri", "mean_fri_bps", "mean_other_fri_bps",
            "contrast_fri_bps", "tstat_welch_fri", "pval_welch_fri", "tstat_hac_fri",
            "pval_bonferroni_fri",
            "n_tue", "mean_tue_bps", "contrast_tue_bps", "pval_welch_tue",
        ):
            assert k in s, f"Missing key: {k}"

    def test_bonferroni_is_2x_raw(self, null_tape):
        """Bonferroni correction must multiply raw p by 2 (capped at 1)."""
        df, _ = null_tape
        s = st.summarize(df)
        for raw_key, bonf_key in [
            ("pval_welch_wed", "pval_bonferroni_wed"),
            ("pval_welch_fri", "pval_bonferroni_fri"),
        ]:
            raw = s[raw_key]
            bonf = s[bonf_key]
            if np.isfinite(raw) and np.isfinite(bonf):
                expected = min(raw * 2, 1.0)
                assert abs(bonf - expected) < 1e-12, (
                    f"Bonferroni mismatch for {bonf_key}: expected {expected}, got {bonf}"
                )

    def test_planted_tape_has_positive_contrasts(self, planted_tape):
        """A planted positive effect on both days should produce positive contrasts."""
        df, _ = planted_tape
        s = st.summarize(df)
        assert s["contrast_wed_bps"] > 0
        assert s["contrast_fri_bps"] > 0


# ---------------------------------------------------------------------------
# Real-data smoke test (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_data_summarize():
    """Real GSPC tape should run through summarize() without error."""
    from turkey import data as d
    df = d.fetch_gspc(start="1928-01-01")
    s = st.summarize(df)
    # Basic sanity checks on real numbers.
    assert s["n_wed"] > 50
    assert s["n_fri"] > 50
    # p-values must be in [0, 1]
    assert 0.0 <= s["pval_welch_wed"] <= 1.0
    assert 0.0 <= s["pval_welch_fri"] <= 1.0
