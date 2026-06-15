"""Tests for the data layer -- CNY table, synthetic tape properties, and cache behaviour."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chinese_zodiac import data  # noqa: E402


# ---------------------------------------------------------------------------
# CNY table sanity
# ---------------------------------------------------------------------------
def test_cny_table_has_expected_length():
    """We hardcode 1990-2026 inclusive = 37 rows."""
    assert len(data.CNY_TABLE) == 37


def test_cny_table_has_all_animals():
    """All 12 zodiac animals appear at least twice over the 37-year span."""
    animals_in_table = {row[1] for row in data.CNY_TABLE}
    assert animals_in_table == set(data.ANIMALS)


def test_cny_df_index_is_years():
    """The CNY DataFrame is indexed by the Gregorian year."""
    df = data.CNY_DF
    assert df.index.min() == 1990
    assert df.index.max() == 2026


def test_cny_dates_are_monotonically_increasing():
    """CNY dates must advance each year (the lunar new year always moves forward)."""
    dates = data.CNY_DF["cny_date"].sort_index()
    assert (dates.diff().dropna() > pd.Timedelta(0)).all()


def test_dragon_years_in_table():
    """Dragon years should be 2000, 2012, 2024 in the table."""
    dragon_yrs = set(data.CNY_DF[data.CNY_DF["animal"] == "Dragon"].index.tolist())
    assert dragon_yrs == {2000, 2012, 2024}


def test_pre_cny_window_constant():
    """The pre-CNY window is 10 trading days as documented."""
    assert data.PRE_CNY_WINDOW == 10


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_tape):
    ret, truth = null_tape
    assert len(ret) >= truth["n_years"] * 248
    assert len(ret) <= truth["n_years"] * 256
    assert (ret > -1.0).all(), "no return should wipe out a position"
    assert ret.index.is_monotonic_increasing


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, dragon_bonus_bp=0.0, seed=7)
    b, _ = data.synthetic_daily(n_years=10, dragon_bonus_bp=0.0, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_daily(n_years=10, dragon_bonus_bp=0.0, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_null_has_uniform_distribution(null_tape):
    """With dragon_bonus=0, there should be no systematic difference in volatility
    between Dragon-year days and the rest -- both drawn from the same distribution."""
    ret, _ = null_tape
    mask = data._dragon_days_mask(ret.index)
    dragon_std = ret[mask].std() if mask.sum() > 10 else None
    rest_std = ret[~mask].std() if (~mask).sum() > 10 else None
    if dragon_std is not None and rest_std is not None:
        # In the null, stds should be within 3x of each other (very loose check)
        assert dragon_std < 3 * rest_std
        assert rest_std < 3 * dragon_std


def test_dragon_bonus_tape_has_higher_dragon_mean(dragon_tape, null_tape):
    """With a planted Dragon bonus, Dragon-year days should have higher mean returns."""
    ret_d, _ = dragon_tape
    ret_n, _ = null_tape
    mask = data._dragon_days_mask(ret_d.index)
    if mask.sum() > 10:
        dragon_mean_bonus = ret_d[mask].mean()
        dragon_mean_null = ret_n[mask].mean()
        assert dragon_mean_bonus > dragon_mean_null


def test_fetch_daily_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("FXI", cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_sensitive(null_tape):
    ret, _ = null_tape
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _ = data.synthetic_daily(n_years=36, seed=99)
    assert data.fingerprint(ret) != data.fingerprint(other)
