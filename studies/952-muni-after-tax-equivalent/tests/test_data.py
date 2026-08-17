"""Data-layer tests — synthetic determinism and the monthly/income decomposition.

Everything here is offline. The only test that touches the shared cache is skipped
outright when the cache is absent, so a fresh checkout with no ``studies/_cache`` is green.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from after_tax import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=952)
    b, _ = data.synthetic_panel(seed=952)
    for key in ("total", "price", "income"):
        assert np.allclose(a[key].to_numpy(), b[key].to_numpy())


def test_synthetic_shape_and_identity(planted):
    panel, truth = planted
    assert set(panel) == {"total", "price", "income"}
    for key in ("total", "price", "income"):
        assert list(panel[key].columns) == ["muni", "taxable", "cash"]
        assert len(panel[key]) == truth["n_months"] == truth["n_years"] * 12
    # total = price + income, by construction and by arithmetic
    assert np.allclose(
        panel["total"].to_numpy(),
        (panel["price"] + panel["income"]).to_numpy(),
    )


def test_synthetic_index_is_periods_and_in_range(planted):
    panel, _ = planted
    idx = panel["total"].index
    assert isinstance(idx, pd.PeriodIndex)
    # OOB-safe: converting to timestamps must stay inside pandas' ns horizon.
    assert idx.to_timestamp()[-1] < pd.Timestamp("2262-01-01")


def test_signal_strength_zero_collapses_the_yield_gap():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=952)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=952)
    assert t1["planted_yield_gap_ann"] > 0.01
    assert abs(t0["planted_yield_gap_ann"]) < 1e-12
    assert abs(t0["planted_breakeven"]) < 1e-12
    assert abs(t1["planted_breakeven"] - 1.0 / 3.0) < 0.01


def test_seed_changes_the_price_legs_but_not_the_coupons():
    a, _ = data.synthetic_panel(seed=952)
    b, _ = data.synthetic_panel(seed=953)
    assert not np.allclose(a["price"].to_numpy(), b["price"].to_numpy())
    assert np.allclose(a["income"].to_numpy(), b["income"].to_numpy())


def test_fingerprint_stable_and_sensitive(planted):
    panel, _ = planted
    fp1 = data.fingerprint(panel["total"])
    other, _ = data.synthetic_panel(seed=953)
    assert fp1 == data.fingerprint(panel["total"]) and len(fp1) == 12
    assert fp1 != data.fingerprint(other["total"])


# --------------------------------------------------------------------------- #
# Monthly aggregation & the income reconstruction
# --------------------------------------------------------------------------- #
def _toy_daily():
    idx = pd.bdate_range("2020-01-01", periods=70)
    total = pd.Series(100.0 * (1.0 + 0.0004) ** np.arange(len(idx)), index=idx)
    price = pd.Series(100.0 * (1.0 + 0.0001) ** np.arange(len(idx)), index=idx)
    return pd.DataFrame({"X": total}), pd.DataFrame({"X": price})


def test_to_monthly_returns_month_end_simple_returns():
    tot, _ = _toy_daily()
    m = data.to_monthly(tot)
    assert isinstance(m.index, pd.PeriodIndex)
    # first month is consumed by the pct_change
    assert len(m) == len(set(tot.index.to_period("M"))) - 1
    per = tot.index.to_period("M")
    last = tot.groupby(per).last()["X"]
    expected = last.iloc[1] / last.iloc[0] - 1.0
    assert abs(m["X"].iloc[0] - expected) < 1e-12


def test_decompose_splits_total_into_price_plus_income():
    tot, pri = _toy_daily()
    panel = data.decompose(tot, pri, floor_income=False)
    assert np.allclose(
        panel["total"]["X"].to_numpy(),
        (panel["price"] + panel["income"])["X"].to_numpy(),
    )
    # the toy tape pays a positive distribution every month
    assert (panel["income"]["X"] > 0).all()


def test_decompose_floor_clips_negative_income():
    idx = pd.bdate_range("2020-01-01", periods=70)
    total = pd.DataFrame({"X": np.linspace(100.0, 101.0, len(idx))}, index=idx)
    price = pd.DataFrame({"X": np.linspace(100.0, 103.0, len(idx))}, index=idx)
    floored = data.decompose(total, price, floor_income=True)["income"]["X"]
    raw = data.decompose(total, price, floor_income=False)["income"]["X"]
    assert (raw < 0).any()
    assert (floored >= 0).all()


def test_load_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_price_only(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (fresh checkout / CI) — "
                           "the synthetic tests cover the logic")
def test_real_cache_race_runs():
    tot = data.load_prices()
    pri = data.load_price_only()
    assert tot.index[-1] <= pd.Timestamp(data.AS_OF)
    panel = data.decompose(tot, pri)
    be = st.breakeven_rate(panel, "MUB", "VCIT")
    r = st.race(panel, "MUB", "VCIT", st.tax_profile(fed_rate=0.37))
    assert np.isfinite(be["breakeven"]) and np.isfinite(r["t_diff"])
    # the taxable corporate leg must out-yield the tax-exempt leg pre-tax
    assert panel["income"]["VCIT"].mean() > panel["income"]["MUB"].mean()
