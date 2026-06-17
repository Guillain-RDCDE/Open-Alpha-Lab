"""Tests for the data layer of Study 271 (Cardboard-Box).

All tests are offline and deterministic -- they use the hardcoded freight table
and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardboard_box import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded freight table
# ---------------------------------------------------------------------------
def test_table_has_expected_rows(freight):
    """1970-2024 inclusive -> 55 rows."""
    assert len(freight) == 55


def test_table_columns(freight):
    expected = {"year", "box_growth", "rail_growth"}
    assert expected.issubset(set(freight.columns))


def test_table_years_sequential(freight):
    assert list(freight["year"]) == list(range(1970, 2025))


def test_table_growth_is_numeric(freight):
    assert pd.api.types.is_numeric_dtype(freight["box_growth"])
    assert pd.api.types.is_numeric_dtype(freight["rail_growth"])


def test_table_recessions_negative(freight):
    """Box growth should be clearly negative in canonical recession years."""
    f = freight.set_index("year")
    for yr in (1975, 2009, 2008):
        assert f.loc[yr, "box_growth"] < 0
        assert f.loc[yr, "rail_growth"] < 0


def test_table_growth_in_plausible_range(freight):
    """YoY growth values stay within a sane band (no typos like 100%)."""
    assert freight["box_growth"].between(-20, 20).all()
    assert freight["rail_growth"].between(-25, 25).all()


def test_freight_table_is_copy(freight):
    t1 = data.freight_table()
    t2 = data.freight_table()
    t1.loc[0, "box_growth"] = 999.0
    assert t2.loc[0, "box_growth"] != 999.0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "box_growth", "fwd_return"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=42)
    b, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=42)
    assert np.allclose(a["fwd_return"].to_numpy(), b["fwd_return"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=1)
    b, _ = data.synthetic_panel(n_years=40, beta=0.0, seed=2)
    assert not np.allclose(a["fwd_return"].to_numpy(), b["fwd_return"].to_numpy())


def test_synthetic_null_no_relationship():
    """beta=0 -> next-year return uncorrelated with the predictor."""
    df, _ = data.synthetic_panel(n_years=4000, beta=0.0, seed=271)
    r = np.corrcoef(df["box_growth"], df["fwd_return"])[0, 1]
    assert abs(r) < 0.06


def test_synthetic_signal_creates_relationship(synthetic_signal):
    """A planted beta -> positive correlation predictor vs next-year return."""
    df, _ = synthetic_signal
    r = np.corrcoef(df["box_growth"], df["fwd_return"])[0, 1]
    assert r > 0.2


# ---------------------------------------------------------------------------
# Forecast frame construction (one-year lag, no look-ahead)
# ---------------------------------------------------------------------------
def test_build_forecast_frame_lags_by_one(freight, synthetic_gspc):
    frame = data.build_forecast_frame(freight, synthetic_gspc, predictor="box_growth")
    # ret_year must be signal_year + 1 everywhere (the execution lag).
    assert (frame["ret_year"] == frame["signal_year"] + 1).all()


def test_build_forecast_frame_columns(freight, synthetic_gspc):
    frame = data.build_forecast_frame(freight, synthetic_gspc)
    assert {"signal_year", "ret_year", "predictor", "fwd_return", "mkt_up"}.issubset(frame.columns)


def test_build_forecast_frame_rail_predictor(freight, synthetic_gspc):
    frame = data.build_forecast_frame(freight, synthetic_gspc, predictor="rail_growth")
    # predictor column equals the rail_growth shifted into place
    merged = frame.merge(
        freight.rename(columns={"year": "signal_year"})[["signal_year", "rail_growth"]],
        on="signal_year",
    )
    assert np.allclose(merged["predictor"], merged["rail_growth"])


# ---------------------------------------------------------------------------
# Cache-only contract & fingerprint
# ---------------------------------------------------------------------------
def test_fetch_gspc_raises_without_cache(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc_annual(fetch=False, cache_path=missing)


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(n_years=30, beta=0.0, seed=10)
    df2, _ = data.synthetic_panel(n_years=30, beta=0.0, seed=11)
    assert data.fingerprint(df1, col="fwd_return") == data.fingerprint(df1, col="fwd_return")
    assert data.fingerprint(df1, col="fwd_return") != data.fingerprint(df2, col="fwd_return")
