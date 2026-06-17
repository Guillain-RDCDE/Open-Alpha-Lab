"""The synthetic tape is well-formed and deterministic; the GSPC loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olympic_year import data  # noqa: E402

GSPC_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "gspc_annual.parquet")
)


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"log_ret", "ret_pct", "is_olympic"}
    assert df.index.name == "year"


def test_synthetic_is_olympic_bool(null_tape):
    df, _ = null_tape
    assert df["is_olympic"].dtype == bool


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=100, olympic_boost=0.0, seed=10)
    b, _ = data.synthetic_annual(n_years=100, olympic_boost=0.0, seed=10)
    assert np.allclose(a["ret_pct"].to_numpy(), b["ret_pct"].to_numpy())
    c, _ = data.synthetic_annual(n_years=100, olympic_boost=0.0, seed=11)
    assert not np.allclose(a["ret_pct"].to_numpy(), c["ret_pct"].to_numpy())


def test_boost_plants_higher_olympic_mean(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    mean_null = df_null[df_null["is_olympic"]]["ret_pct"].mean()
    mean_boost = df_boost[df_boost["is_olympic"]]["ret_pct"].mean()
    # The boosted tape should have a meaningfully higher Olympic-year mean.
    assert mean_boost > mean_null + 5.0  # at least 5 pp higher


def test_boost_does_not_affect_non_olympic(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    # Non-Olympic years should have the same means (same seed, same noise for those years).
    m_null = df_null[~df_null["is_olympic"]]["ret_pct"].mean()
    m_boost = df_boost[~df_boost["is_olympic"]]["ret_pct"].mean()
    assert abs(m_null - m_boost) < 1e-8


def test_olympic_years_present_in_synthetic(null_tape):
    df, truth = null_tape
    assert df["is_olympic"].any(), "At least some Olympic years should appear in the tape"
    assert df["is_olympic"].sum() >= 1


def test_fingerprint_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    df2, _ = data.synthetic_annual(n_years=100, olympic_boost=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(df2)


@pytest.mark.skipif(not os.path.exists(GSPC_CACHE), reason="real-tape cache absent offline/CI")
def test_gspc_loader_returns_correct_columns():
    df = data.load_gspc(cache_path=GSPC_CACHE)
    assert set(df.columns) >= {"log_ret", "ret_pct", "is_olympic"}
    assert df.index.name == "year"


@pytest.mark.skipif(not os.path.exists(GSPC_CACHE), reason="real-tape cache absent offline/CI")
def test_gspc_loader_olympic_flag_correct():
    from olympic_year.data import OLYMPIC_YEARS
    df = data.load_gspc(cache_path=GSPC_CACHE)
    for year in df.index:
        expected = year in OLYMPIC_YEARS
        assert df.loc[year, "is_olympic"] == expected


def test_gspc_loader_raises_without_cache(tmp_path):
    """Without a local cache and no network, load_gspc raises an error gracefully.

    In the offline / CI environment, this tests the cache-miss path without triggering
    a network call.  We point at a non-existent cache file and intercept the error at
    the parquet read stage (since we cannot intercept yfinance in the test environment).
    The function must either raise FileNotFoundError or a pandas/pyarrow read error —
    either is acceptable: what matters is that it does NOT silently return bad data.
    """
    import pytest
    bad_cache = str(tmp_path / "no_such_file.parquet")
    # Patch the fetch function to avoid any network attempt.
    import unittest.mock as mock
    with mock.patch("olympic_year.data._fetch_and_cache", side_effect=FileNotFoundError("mocked")):
        with pytest.raises((FileNotFoundError, Exception)):
            data.load_gspc(cache_path=bad_cache)
