"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tax_day import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tape — shape, determinism, calendar correctness
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert set(df.columns) >= {"ret", "pre_tax", "post_tax", "tax_month"}
    assert len(df) == truth["n_days"]
    assert df["ret"].notna().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=20, pre_tax_effect=0.0, seed=5)
    b, _ = data.synthetic_daily(n_years=20, pre_tax_effect=0.0, seed=5)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(n_years=20, pre_tax_effect=0.0, seed=6)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_synthetic_pre_post_mutually_exclusive(null_tape):
    df, _ = null_tape
    # A day cannot be simultaneously in both windows.
    overlap = df["pre_tax"] & df["post_tax"]
    assert not overlap.any(), "pre_tax and post_tax must not overlap"


def test_tax_window_counts_reasonable(null_tape):
    df, truth = null_tape
    # ~10 pre-tax trading days per year * 75 years = ~750 (allow +/- 20%).
    n_pre = int(df["pre_tax"].sum())
    assert 500 <= n_pre <= 900, f"unexpected n_pre={n_pre}"
    # ~5 post-tax days per year.
    n_post = int(df["post_tax"].sum())
    assert 250 <= n_post <= 500, f"unexpected n_post={n_post}"


def test_tax_windows_only_in_april(null_tape):
    df, _ = null_tape
    # Pre-tax window should fall in March/April (near-deadline days in March are possible
    # if the business calendar runs across month ends late).
    pre_months = set(df.index[df["pre_tax"]].month)
    assert pre_months <= {3, 4}, f"unexpected pre_tax months: {pre_months}"
    # Post-tax window: on or after April 15, so always in April (or occasionally May if the
    # post window straddles the month boundary).
    post_months = set(df.index[df["post_tax"]].month)
    assert post_months <= {4, 5}, f"unexpected post_tax months: {post_months}"


def test_effective_tax_deadline_shifts_weekend():
    # April 15, 2023 was a Saturday; deadline should shift to Monday April 17.
    d = data._effective_tax_deadline(2023)
    assert d == pd.Timestamp("2023-04-17"), f"got {d}"
    # April 15, 2018 was a Sunday; deadline should shift to Monday April 16.
    d2 = data._effective_tax_deadline(2018)
    assert d2 == pd.Timestamp("2018-04-16"), f"got {d2}"
    # April 15, 2024 was a Monday; no shift needed.
    d3 = data._effective_tax_deadline(2024)
    assert d3 == pd.Timestamp("2024-04-15"), f"got {d3}"


def test_effect_knob_shifts_mean_in_window(pre_effect_tape, null_tape):
    """When a pre-tax effect is planted, the window mean is higher than on the null tape."""
    df_eff, _ = pre_effect_tape
    df_null, _ = null_tape
    mean_eff = df_eff.loc[df_eff["pre_tax"], "ret"].mean()
    mean_null = df_null.loc[df_null["pre_tax"], "ret"].mean()
    assert mean_eff > mean_null, "planted positive pre-tax effect must raise window mean"


def test_fingerprint_stable_and_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_years=75, pre_tax_effect=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without any cache files the loader must raise FileNotFoundError, not hang."""
    # Temporarily override the module-level paths.
    orig_shared = data._GSPC_SHARED
    orig_local = data._GSPC_LOCAL
    try:
        data._GSPC_SHARED = str(tmp_path / "nonexistent_shared.parquet")
        data._GSPC_LOCAL = str(tmp_path / "nonexistent_local.parquet")
        with pytest.raises(FileNotFoundError):
            data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))
    finally:
        data._GSPC_SHARED = orig_shared
        data._GSPC_LOCAL = orig_local


# ---------------------------------------------------------------------------
# Real-data tests (skipped in CI when cache is absent)
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402 (needed by test_effective_tax_deadline_shifts_weekend above)

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
def test_real_gspc_shape_and_range():
    df = data.fetch_gspc(start="1928-01-01")
    assert len(df) > 20_000, "expect many decades of daily data"
    assert df.index[0].year <= 1928 or df.index[0].year <= 1930
    assert set(df.columns) >= {"ret", "pre_tax", "post_tax", "tax_month"}


@requires_cache
def test_real_gspc_tax_windows_sensible():
    df = data.fetch_gspc(start="1928-01-01")
    # Should have ~1 pre-tax window (10 days) per year.
    years = df.index.year.max() - df.index.year.min() + 1
    n_pre = int(df["pre_tax"].sum())
    # Allow substantial slack for missing April data in early years.
    assert n_pre > years * 5, f"too few pre_tax days: {n_pre} for {years} years"
