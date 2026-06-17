"""The synthetic world is deterministic; the winter premium is recovered when planted; the null shows
nothing; the seasonal strategy beats buy-and-hold only when the premium is real; the signal correctly
marks Oct-Mar as winter; and a gapped cache fails loudly. Offline, seeded."""

import os

import numpy as np
import pandas as pd
import pytest

from natgas_winter import data, strategy as st

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "natgas_winter_ung.parquet",
)


def test_world_deterministic(premium_world):
    df, truth = premium_world
    df2, _ = data.synthetic_world(winter_premium=0.04, seed=227)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_premium_world_significant_tstat(premium_world):
    """With a planted 4%/month winter premium, the t-stat must be significant."""
    df, _ = premium_world
    result = st.winter_premium(df["ung"])
    # planted premium => slope positive and significant
    assert result["slope"] > 0.0
    assert result["tstat"] > 2.0


def test_null_world_no_premium(null_world):
    """With no planted premium, |t| must stay below 2."""
    df, _ = null_world
    result = st.winter_premium(df["ung"])
    assert abs(result["tstat"]) < 2.0


def test_seasonal_beats_buyhold_when_premium_real(premium_world):
    """When winter genuinely outperforms, the seasonal strategy Sharpe exceeds buy-and-hold."""
    df, _ = premium_world
    seasonal = st.summary(st.seasonal_returns(df["ung"]))
    bh = st.summary(st.buy_hold_returns(df["ung"]))
    assert seasonal["sharpe"] > bh["sharpe"]


def test_seasonal_signal_marks_oct_mar():
    """Oct (10), Nov (11), Dec (12), Jan (1), Feb (2), Mar (3) must be flagged winter."""
    idx = pd.date_range("2020-09-30", periods=12, freq="ME")
    sig = st.seasonal_signal(pd.DatetimeIndex(idx))
    months = idx.month
    for i, m in enumerate(months):
        expected = 1.0 if m in st.WINTER_MONTHS else 0.0
        assert sig.iloc[i] == expected, f"month {m} expected {expected}, got {sig.iloc[i]}"


def test_seasonal_returns_zero_outside_winter(null_world):
    """The seasonal strategy returns 0.0 in non-winter months."""
    df, _ = null_world
    seasonal = st.seasonal_returns(df["ung"])
    non_winter = pd.DatetimeIndex(seasonal.index).month.map(lambda m: m not in st.WINTER_MONTHS)
    assert (seasonal[non_winter] == 0.0).all()


@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_continuous():
    """The cached real data must have no interior monthly holes."""
    d = data.fetch_ung()
    months = pd.PeriodIndex(d.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    assert len(expected.difference(months)) == 0


def test_gapped_cache_fails_loudly(tmp_path, premium_world):
    """A stale cache with interior holes must fail loudly, not silently mis-date."""
    df, _ = premium_world
    gapped = df.drop(df.index[5:8])
    gapped.to_parquet(tmp_path / "natgas_winter_ung.parquet")
    with pytest.raises(AssertionError, match="hole"):
        data.fetch_ung(cache_dir=str(tmp_path))


def test_continuous_cache_accepted(tmp_path, premium_world):
    """A correctly-built cache with no holes is read back cleanly."""
    df, _ = premium_world
    df.to_parquet(tmp_path / "natgas_winter_ung.parquet")
    out = data.fetch_ung(cache_dir=str(tmp_path))
    assert list(out.columns) == ["ung"]
    assert len(out) == len(df)
