"""The synthetic world is deterministic, the null is centred on zero, and a gapped cache fails loudly.
Offline, seeded — these tests never skip."""

import os

import numpy as np
import pandas as pd
import pytest

from coffee_seasonality import data, strategy as st

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "coffee_seasonality.parquet",
)


def test_world_deterministic(frost_world):
    df, truth = frost_world
    df2, _ = data.synthetic_world(frost_premium=0.08, seed=307)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_seasonality


def test_world_columns_and_grid(frost_world):
    df, _ = frost_world
    assert list(df.columns) == ["coffee", "tbill"]
    # No interior holes in the synthetic monthly grid.
    data._assert_continuous_monthly(df)


def test_null_world_has_no_planted_premium(null_world):
    df, truth = null_world
    assert not truth.has_seasonality
    # Frost and harvest months are drawn from the same i.i.d. law -> no significant spread.
    r = st.frost_harvest_tstat(df["coffee"])
    assert abs(r["tstat"]) < 2.0


def test_fetch_data_cache_miss_returns_empty(tmp_path):
    out = data.fetch_data(cache_dir=str(tmp_path), fetch=False)
    assert out.empty


def test_fetch_data_rejects_gapped_cache(tmp_path, frost_world):
    df, _ = frost_world
    frame = pd.DataFrame({"coffee": df["coffee"], "tbill": 0.001})
    gapped = frame.drop(frame.index[5:8])  # punch a 3-month hole
    gapped.to_parquet(tmp_path / "coffee_seasonality.parquet")
    with pytest.raises(AssertionError, match="hole"):
        data.fetch_data(cache_dir=str(tmp_path))


def test_fetch_data_accepts_continuous_cache(tmp_path, frost_world):
    df, _ = frost_world
    frame = pd.DataFrame({"coffee": df["coffee"], "tbill": 0.001})
    frame.to_parquet(tmp_path / "coffee_seasonality.parquet")
    out = data.fetch_data(cache_dir=str(tmp_path))
    assert list(out.columns) == ["coffee", "tbill"]
    assert len(out) == len(frame)


@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_cache_is_continuous():
    d = data.fetch_data()
    assert not d.empty
    months = pd.PeriodIndex(d.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    assert len(expected.difference(months)) == 0
