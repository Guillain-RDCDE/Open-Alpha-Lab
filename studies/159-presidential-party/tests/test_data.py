"""The PRESIDENTS table, party_label, and the synthetic generator are self-consistent."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidential_party import data  # noqa: E402


def test_presidents_table_is_chronological():
    """Entries in PRESIDENTS are in ascending time order with no gaps or overlaps."""
    records = data._PRES_RECORDS
    for i in range(1, len(records)):
        prev_end = records[i - 1]["end"]
        curr_start = records[i]["start"]
        # Adjacent terms: start of next is one day or one month after end of prev
        import pandas as pd
        gap = (curr_start - prev_end).days
        assert 0 <= gap <= 32, (
            f"Gap/overlap between {records[i-1]['name']} and {records[i]['name']}: "
            f"{gap} days"
        )


def test_presidents_table_parties_are_d_or_r():
    for rec in data._PRES_RECORDS:
        assert rec["party"] in ("D", "R"), f"Unknown party for {rec['name']}: {rec['party']}"


def test_party_label_covers_known_dates():
    import pandas as pd

    idx = pd.DatetimeIndex(["1945-04-15", "1953-05-01", "1993-03-01", "2009-03-01"])
    labels = data.party_label(idx)
    # Truman (April 1945): D
    assert labels.iloc[0] == "D"
    # Eisenhower (May 1953): R
    assert labels.iloc[1] == "R"
    # Clinton (March 1993): D
    assert labels.iloc[2] == "D"
    # Obama (March 2009): D
    assert labels.iloc[3] == "D"


def test_party_label_outside_coverage_is_nan():
    import pandas as pd

    idx = pd.DatetimeIndex(["1900-01-01", "2050-01-01"])
    labels = data.party_label(idx)
    assert labels.isna().all()


def test_synthetic_is_deterministic(null_tape):
    df, truth = null_tape
    df2, _ = data.synthetic_monthly(n_months=truth["n_months"],
                                     dem_premium_bps=truth["dem_premium_bps"],
                                     seed=truth["seed"])
    assert np.allclose(df["ret"].values, df2["ret"].values)


def test_synthetic_null_has_no_party_premium(null_tape):
    """With dem_premium = 0, D and R monthly means should be within noise of each other."""
    df, _ = null_tape
    dem_mean = df.loc[df["party"] == "D", "ret"].mean()
    rep_mean = df.loc[df["party"] == "R", "ret"].mean()
    # On n=1200 months the noise is small; with zero premium the gap should be < 30 bps/month
    assert abs(dem_mean - rep_mean) < 0.003, (
        f"Null tape has too large a D-R gap: {(dem_mean - rep_mean) * 1e4:.1f} bps/month"
    )


def test_synthetic_planted_premium_is_recoverable():
    """With a large dem_premium planted (500 bps/month), D mean must exceed R mean."""
    # Use 500 bps/month so signal-to-noise is decisive on n=1200 months
    df, _ = data.synthetic_monthly(n_months=1200, dem_premium_bps=500.0, seed=159)
    dem_mean = df.loc[df["party"] == "D", "ret"].mean()
    rep_mean = df.loc[df["party"] == "R", "ret"].mean()
    gap_bps = (dem_mean - rep_mean) * 1e4
    # Planted 500 bps/month — on n~600 months per party the noise SE ~19 bps
    # so gap should easily exceed 200 bps
    assert gap_bps > 200, (
        f"Large planted premium not recovered: D-R gap = {gap_bps:.1f} bps/month (expected > 200)"
    )


def test_synthetic_party_coverage(null_tape):
    """Both parties are present and cover a substantial fraction of months."""
    df, _ = null_tape
    n_d = (df["party"] == "D").sum()
    n_r = (df["party"] == "R").sum()
    n_total = len(df)
    assert n_d > 0 and n_r > 0
    # Alternating 48-month terms: each party gets ~50% of months
    assert n_d / n_total > 0.40
    assert n_r / n_total > 0.40


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_monthly(n_months=1200, dem_premium_bps=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_shiller_cache_path_exists():
    """The shared Shiller cache must exist (pre-staged by the repo)."""
    assert os.path.exists(data._SHILLER_CACHE), (
        f"Shiller cache not found at {data._SHILLER_CACHE}. "
        "Expected pre-staged by repo (_cache/shiller_sp500.parquet)."
    )


def test_load_shiller_returns_party_labelled_df():
    """load_shiller returns a DataFrame with ret and party columns, no NaN parties."""
    df = data.load_shiller()
    assert "ret" in df.columns
    assert "party" in df.columns
    assert df["party"].notna().all()
    assert set(df["party"].unique()).issubset({"D", "R"})
    assert len(df) > 500, f"Expected > 500 months of data, got {len(df)}"


def test_load_shiller_has_both_parties():
    df = data.load_shiller()
    assert "D" in df["party"].values
    assert "R" in df["party"].values


def test_gspc_cache_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(cache_dir=str(tmp_path), fetch=False)
