"""Data tests: synthetic tape shapes, calendar arithmetic, and cache-gating.

The synthetic tests run fully offline. Real-data tests are guarded with
``requires_cache`` and skip when the shared ^GSPC parquet is absent (CI)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turkey import data  # noqa: E402

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
# Calendar arithmetic
# ---------------------------------------------------------------------------
class TestThanksgivingDates:
    def test_thanksgiving_is_fourth_thursday_of_november(self):
        """_thanksgiving_year must return the 4th Thursday of November."""
        for year in range(1950, 2030):
            t = data._thanksgiving_year(year)
            assert t.month == 11, f"Thanksgiving not in November for {year}"
            assert t.weekday() == 3, f"Thanksgiving not a Thursday for {year}"
            # Must be the 4th Thursday: day in [22, 28]
            assert 22 <= t.day <= 28, f"Thanksgiving day {t.day} not in [22,28] for {year}"

    def test_known_thanksgiving_dates(self):
        """Spot-check known US Thanksgiving dates."""
        known = {
            2000: pd.Timestamp("2000-11-23"),
            2010: pd.Timestamp("2010-11-25"),
            2020: pd.Timestamp("2020-11-26"),
            2024: pd.Timestamp("2024-11-28"),
        }
        for year, expected in known.items():
            assert data._thanksgiving_year(year) == expected, (
                f"Wrong Thanksgiving for {year}: got {data._thanksgiving_year(year)}"
            )

    def test_wed_before_is_one_day_before_thursday(self):
        """The Wednesday before Thanksgiving must be Thanksgiving - 1 day."""
        for year in range(2000, 2025):
            t = data._thanksgiving_year(year)
            expected_wed = t - pd.Timedelta(days=1)
            assert expected_wed.weekday() == 2, f"Wed-before not a Wednesday in {year}"

    def test_fri_after_is_one_day_after_thursday(self):
        """The Friday after Thanksgiving must be Thanksgiving + 1 day."""
        for year in range(2000, 2025):
            t = data._thanksgiving_year(year)
            expected_fri = t + pd.Timedelta(days=1)
            assert expected_fri.weekday() == 4, f"Fri-after not a Friday in {year}"

    def test_placebo_tue_is_two_days_before_thursday(self):
        """The Tuesday placebo must be Thanksgiving - 2 days."""
        for year in range(2000, 2025):
            t = data._thanksgiving_year(year)
            expected_tue = t - pd.Timedelta(days=2)
            assert expected_tue.weekday() == 1, f"Placebo-Tue not a Tuesday in {year}"


class TestMasks:
    def setup_method(self):
        idx = pd.bdate_range("1950-01-04", periods=75 * 252)
        self.idx = idx

    def test_wed_mask_count(self):
        """Should have roughly 75 Thanksgiving Wednesdays (one per year)."""
        mask = data.is_wed_before_thanksgiving(self.idx)
        # Should be approximately 75 (some years the Wed may be a holiday, but bdate_range
        # excludes weekends not holidays, so count should be close to 75).
        assert 60 <= mask.sum() <= 80

    def test_fri_mask_count(self):
        """Should have roughly 75 Black Fridays."""
        mask = data.is_fri_after_thanksgiving(self.idx)
        assert 60 <= mask.sum() <= 80

    def test_no_overlap_between_wed_and_fri(self):
        """Wednesday-before and Friday-after cannot overlap."""
        wed = data.is_wed_before_thanksgiving(self.idx)
        fri = data.is_fri_after_thanksgiving(self.idx)
        assert (wed & fri).sum() == 0

    def test_placebo_tue_and_wed_no_overlap(self):
        """Tuesday placebo and Wednesday-before cannot overlap."""
        tue = data.is_placebo_tue(self.idx)
        wed = data.is_wed_before_thanksgiving(self.idx)
        assert (tue & wed).sum() == 0

    def test_masks_are_boolean_series(self):
        wed = data.is_wed_before_thanksgiving(self.idx)
        fri = data.is_fri_after_thanksgiving(self.idx)
        assert wed.dtype == bool
        assert fri.dtype == bool


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
class TestSyntheticDaily:
    def test_returns_tuple(self, null_tape):
        df, truth = null_tape
        assert isinstance(df, pd.DataFrame)
        assert isinstance(truth, dict)

    def test_columns(self, null_tape):
        df, _ = null_tape
        for col in ("ret", "is_wed_before_tsg", "is_fri_after_tsg", "is_placebo_tue"):
            assert col in df.columns

    def test_no_nans_in_ret(self, null_tape):
        df, _ = null_tape
        assert df["ret"].isna().sum() == 0

    def test_truth_keys(self, null_tape):
        _, truth = null_tape
        for k in ("n_years", "annual_vol", "annual_drift", "wed_effect",
                   "fri_effect", "n_days", "n_wed", "n_fri", "seed"):
            assert k in truth

    def test_n_wed_positive(self, null_tape):
        _, truth = null_tape
        assert truth["n_wed"] > 0

    def test_n_fri_positive(self, null_tape):
        _, truth = null_tape
        assert truth["n_fri"] > 0

    def test_planted_effect_shifts_wed_mean(self):
        """A strongly planted positive effect should raise the Wednesday-before mean."""
        df_null, _ = data.synthetic_daily(n_years=75, wed_effect=0.0, seed=194)
        df_pos, _ = data.synthetic_daily(n_years=75, wed_effect=5.0, seed=194)
        # The mean return on Wed-before should be clearly higher in the planted tape.
        mean_null = df_null.loc[df_null["is_wed_before_tsg"], "ret"].mean()
        mean_pos = df_pos.loc[df_pos["is_wed_before_tsg"], "ret"].mean()
        assert mean_pos > mean_null + 0.001  # at least 10 bps

    def test_planted_effect_shifts_fri_mean(self):
        """A strongly planted positive effect should raise the Black Friday mean."""
        df_null, _ = data.synthetic_daily(n_years=75, fri_effect=0.0, seed=194)
        df_pos, _ = data.synthetic_daily(n_years=75, fri_effect=5.0, seed=194)
        mean_null = df_null.loc[df_null["is_fri_after_tsg"], "ret"].mean()
        mean_pos = df_pos.loc[df_pos["is_fri_after_tsg"], "ret"].mean()
        assert mean_pos > mean_null + 0.001

    def test_deterministic(self):
        """Same seed must produce identical tapes."""
        df1, _ = data.synthetic_daily(n_years=50, seed=42)
        df2, _ = data.synthetic_daily(n_years=50, seed=42)
        np.testing.assert_array_equal(df1["ret"].values, df2["ret"].values)

    def test_different_seeds_differ(self):
        """Different seeds must produce different tapes."""
        df1, _ = data.synthetic_daily(n_years=50, seed=1)
        df2, _ = data.synthetic_daily(n_years=50, seed=2)
        assert not np.allclose(df1["ret"].values, df2["ret"].values)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
class TestFingerprint:
    def test_fingerprint_is_string(self, null_tape):
        df, _ = null_tape
        fp = data.fingerprint(df)
        assert isinstance(fp, str) and len(fp) == 12

    def test_fingerprint_stable(self, null_tape):
        df, _ = null_tape
        assert data.fingerprint(df) == data.fingerprint(df)


# ---------------------------------------------------------------------------
# Real-data tests (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_fetch_gspc_columns():
    """Real GSPC tape should have the right columns."""
    df = data.fetch_gspc(start="1950-01-01")
    for col in ("close", "ret", "is_wed_before_tsg", "is_fri_after_tsg", "is_placebo_tue"):
        assert col in df.columns


@requires_cache
def test_fetch_gspc_n_wed_positive():
    """Real tape should have positive counts of Thanksgiving Wednesdays."""
    df = data.fetch_gspc(start="1950-01-01")
    assert df["is_wed_before_tsg"].sum() > 0


@requires_cache
def test_fetch_gspc_n_fri_positive():
    """Real tape should have positive counts of Black Fridays."""
    df = data.fetch_gspc(start="1950-01-01")
    assert df["is_fri_after_tsg"].sum() > 0


@requires_cache
def test_fetch_gspc_no_overlap():
    """No date should be both a Wednesday-before and Friday-after Thanksgiving."""
    df = data.fetch_gspc(start="1950-01-01")
    overlap = df["is_wed_before_tsg"] & df["is_fri_after_tsg"]
    assert overlap.sum() == 0


@requires_cache
def test_fingerprint_real():
    """Real data fingerprint should be a 12-char hex string."""
    df = data.fetch_gspc(start="1950-01-01")
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12
