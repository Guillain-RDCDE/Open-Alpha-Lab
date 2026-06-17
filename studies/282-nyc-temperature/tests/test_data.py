"""Tests for the data layer of Study 282 (NYC-Temperature).

All tests are offline and deterministic -- the curated temperature table and the
synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyc_temperature import data  # noqa: E402


# ---------------------------------------------------------------------------
# Curated monthly anomaly table
# ---------------------------------------------------------------------------
def test_table_rows_and_columns(anomaly_table):
    """64 years x 12 months = 768 rows, with the expected columns."""
    assert len(anomaly_table) == (data.YEAR_END - data.YEAR_START + 1) * 12
    assert set(anomaly_table.columns) == {"year", "month", "anomaly_c"}


def test_table_year_range(anomaly_table):
    assert anomaly_table["year"].min() == 1962
    assert anomaly_table["year"].max() == 2025


def test_table_months_complete(anomaly_table):
    assert set(anomaly_table["month"].unique()) == set(range(1, 13))


def test_table_anomalies_plausible(anomaly_table):
    """NYC monthly anomalies should be modest (between -3 and +3 C)."""
    assert anomaly_table["anomaly_c"].between(-3.0, 3.0).all()


def test_table_has_warming_trend(anomaly_table):
    """Recent decades should be warmer than the 1960s on average."""
    early = anomaly_table.loc[anomaly_table["year"] <= 1971, "anomaly_c"].mean()
    late = anomaly_table.loc[anomaly_table["year"] >= 2016, "anomaly_c"].mean()
    assert late > early + 0.5


# ---------------------------------------------------------------------------
# Daily temperature builder
# ---------------------------------------------------------------------------
def test_build_daily_temperature_columns():
    dates = pd.bdate_range("2010-01-01", periods=500)
    t = data.build_daily_temperature(dates, seed=282)
    assert set(t.columns) == {"temp_f", "temp_normal_f", "temp_anomaly_f"}
    assert len(t) == 500


def test_anomaly_is_temp_minus_normal():
    dates = pd.bdate_range("2010-01-01", periods=300)
    t = data.build_daily_temperature(dates, seed=282)
    diff = t["temp_f"] - t["temp_normal_f"]
    assert np.allclose(diff.to_numpy(), t["temp_anomaly_f"].to_numpy())


def test_seasonal_cycle_present():
    """July should be warmer than January in the normal curve."""
    dates = pd.bdate_range("2015-01-01", "2015-12-31")
    t = data.build_daily_temperature(dates, seed=282)
    jan = t.loc[t.index.month == 1, "temp_normal_f"].mean()
    jul = t.loc[t.index.month == 7, "temp_normal_f"].mean()
    assert jul > jan + 20.0


def test_build_daily_deterministic():
    dates = pd.bdate_range("2010-01-01", periods=400)
    a = data.build_daily_temperature(dates, seed=282)
    b = data.build_daily_temperature(dates, seed=282)
    assert np.allclose(a["temp_f"].to_numpy(), b["temp_f"].to_numpy())


# ---------------------------------------------------------------------------
# Synthetic daily generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_days"]
    assert set(df.columns) >= {"ret", "temp_anomaly_f", "temp_f", "temp_normal_f"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_null_has_no_return_temp_correlation():
    """Under the null, returns are ~uncorrelated with the temperature anomaly."""
    df, _ = data.synthetic_daily(n_days=8000, premium_bps=0.0, seed=282)
    corr = np.corrcoef(df["ret"].to_numpy(), df["temp_anomaly_f"].to_numpy())[0, 1]
    assert abs(corr) < 0.05


def test_planted_premium_creates_correlation():
    """A large planted cold premium makes cold days earn more (negative corr)."""
    df, _ = data.synthetic_daily(n_days=8000, premium_bps=60.0, seed=282)
    corr = np.corrcoef(df["ret"].to_numpy(), df["temp_anomaly_f"].to_numpy())[0, 1]
    # Cold (negative anomaly) earns more -> negative correlation.
    assert corr < -0.1


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=10)
    df2, _ = data.synthetic_daily(n_days=500, premium_bps=0.0, seed=11)
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


def test_fetch_gspc_raises_without_cache(tmp_path):
    """Without the cache, fetch_gspc_daily(fetch=False) raises FileNotFoundError."""
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_daily(fetch=False, cache_path=missing)
