"""The synthetic tape is well-formed and deterministic; the Shiller loader is
cache-safe and returns sensible real returns for the 1881-2023 window."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hundred_minus_age import data  # noqa: E402


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_lifecycle(n_years=40, seed=172)
    assert list(prices.columns) == ["EQ", "BD"]
    assert len(prices) == 40 * 12
    assert (prices > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_lifecycle(n_years=20, equity_premium=0.04, seed=5)
    b, _ = data.synthetic_lifecycle(n_years=20, equity_premium=0.04, seed=5)
    assert np.allclose(a["EQ"].to_numpy(), b["EQ"].to_numpy())
    c, _ = data.synthetic_lifecycle(n_years=20, equity_premium=0.04, seed=6)
    assert not np.allclose(a["EQ"].to_numpy(), c["EQ"].to_numpy())


def test_synthetic_premium_shifts_equity_mean():
    """A higher equity_premium should raise the mean equity return."""
    _, _ = data.synthetic_lifecycle(n_years=40, equity_premium=0.0, seed=172)
    p_no, _ = data.synthetic_lifecycle(n_years=40, equity_premium=0.0, seed=172)
    p_yes, _ = data.synthetic_lifecycle(n_years=40, equity_premium=0.04, seed=172)
    ret_no = p_no["EQ"].pct_change().dropna()
    ret_yes = p_yes["EQ"].pct_change().dropna()
    assert ret_yes.mean() > ret_no.mean()


def test_synthetic_null_equity_and_bond_same_mean():
    """With equity_premium=0 the equity and bond means should be equal."""
    prices, truth = data.synthetic_lifecycle(n_years=80, equity_premium=0.0, seed=172)
    ret = prices.pct_change().dropna()
    assert abs(ret["EQ"].mean() - ret["BD"].mean()) < 0.0015


def test_fingerprint_stable_and_sensitive():
    prices_a, _ = data.synthetic_lifecycle(n_years=20, seed=172)
    ret_a = prices_a.pct_change().dropna()
    prices_b, _ = data.synthetic_lifecycle(n_years=20, seed=172)
    ret_b = prices_b.pct_change().dropna()
    assert data.fingerprint(ret_a) == data.fingerprint(ret_b)
    prices_c, _ = data.synthetic_lifecycle(n_years=20, seed=99)
    ret_c = prices_c.pct_change().dropna()
    assert data.fingerprint(ret_a) != data.fingerprint(ret_c)


def test_shiller_loader_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_shiller(cache_path=str(tmp_path / "nope.parquet"))


def test_shiller_returns_sensible_values():
    """Equity real annual return ~6-8%; bond real annual return ~3-5% historically."""
    ret = data.load_shiller()
    eq_ann = (1.0 + ret["EQ"].mean()) ** 12 - 1.0
    bd_ann = (1.0 + ret["BD"].mean()) ** 12 - 1.0
    assert 0.04 < eq_ann < 0.12, f"Unexpected EQ annual real return: {eq_ann:.3f}"
    assert 0.01 < bd_ann < 0.08, f"Unexpected BD annual real return: {bd_ann:.3f}"
    assert eq_ann > bd_ann, "Equities should earn more than bonds over 1881-2023"


def test_shiller_covers_enough_history():
    """The real tape must cover at least 100 years to give meaningful cohort counts."""
    ret = data.load_shiller()
    n_years = len(ret) / 12.0
    assert n_years > 100, f"Only {n_years:.1f} years of Shiller data"


def test_shiller_no_nan_in_core_columns():
    ret = data.load_shiller()
    assert not ret["EQ"].isna().any()
    assert not ret["BD"].isna().any()
