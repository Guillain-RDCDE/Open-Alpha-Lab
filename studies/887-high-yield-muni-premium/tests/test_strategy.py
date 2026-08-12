"""Offline, fixed-seed tests for the HY-muni-premium machinery.

The synthetic world is deterministic; the HAC/bootstrap pipeline recovers a planted
credit premium and stays quiet on the null; the tax-equivalent-yield and after-tax
arithmetic are correct; costs reduce the net; the era/drawdown helpers behave; no
look-ahead in the monthly-return alignment. All offline — no network, no real cache.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_muni import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, planted premium, null
# --------------------------------------------------------------------------- #
def test_world_deterministic(premium_world):
    w2 = data.synthetic_world(premium_annual=0.03, seed=887, n_months=200)
    assert np.allclose(premium_world.to_numpy(), w2.to_numpy())


def test_world_index_no_overflow(premium_world):
    # PeriodIndex-derived timestamps must be well inside the pandas ns horizon (year 2262).
    assert premium_world.index.max() < pd.Timestamp("2100-01-01")
    assert premium_world.index.is_monotonic_increasing


def test_planted_premium_recovered(premium_world):
    sp = st.premium_series(premium_world)
    h = st.hac_mean(sp.values)
    boot = st.bootstrap_mean_ci(sp.values)
    assert h["tstat"] > 3.0                    # planted premium lights up
    assert h["mean_bps"] > 0
    assert boot["ci_low_bps"] > 0              # bootstrap CI clear of zero
    # planted 3%/yr ~ 25 bps/mo; recovered mean should be in the right ballpark
    assert 15.0 < h["mean_bps"] < 40.0


def test_null_world_no_signal(null_world):
    sp = st.premium_series(null_world)
    h = st.hac_mean(sp.values)
    boot = st.bootstrap_mean_ci(sp.values)
    assert abs(h["tstat"]) < 2.0               # does not manufacture significance
    assert boot["ci_low_bps"] < 0 < boot["ci_high_bps"]   # CI straddles zero


def test_synthetic_detect_keys(premium_world):
    d = st.synthetic_detect(premium_world)
    assert set(d) >= {"mean_bps", "tstat", "n", "ci_low_bps", "ci_high_bps", "frac_negative"}
    assert d["n"] == 200


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_hac_mean_matches_plain_t_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.02, 3000)
    plain_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    hac_t = st.hac_mean(x)["tstat"]
    assert abs(hac_t - plain_t) < 0.6          # HAC ~ plain when serially uncorrelated


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0020, 0.01, 1500)
    b = st.bootstrap_mean_ci(x, n_boot=2000, seed=5)
    assert b["ci_low_bps"] < b["mean_bps"] < b["ci_high_bps"]


def test_bootstrap_deterministic():
    rng = np.random.default_rng(2)
    x = rng.normal(0.001, 0.01, 500)
    a = st.bootstrap_mean_ci(x, seed=42)
    b = st.bootstrap_mean_ci(x, seed=42)
    assert a["ci_low_bps"] == b["ci_low_bps"] and a["ci_high_bps"] == b["ci_high_bps"]


# --------------------------------------------------------------------------- #
# The tax wrapper
# --------------------------------------------------------------------------- #
def test_tax_equivalent_yield_identity():
    # y / (1 - t): a 5% tax-exempt yield at 40% tax equals an 8.33% taxable yield.
    assert abs(st.tax_equivalent_yield(5.0, 0.40) - 8.3333) < 1e-3
    # a muni's TEY always exceeds its raw yield for any positive tax rate
    assert st.tax_equivalent_yield(4.0, 0.408) > 4.0


def test_after_tax_muni_keeps_income():
    idx = pd.period_range("2015-01", periods=24, freq="M").to_timestamp(how="end")
    tr = pd.DataFrame({"X": np.full(24, 0.008), "BIL": np.full(24, 0.001)}, index=idx)
    income = pd.DataFrame({"X": np.full(24, 0.003)}, index=idx)   # 3 of the 8 is coupon
    at_exempt = st.after_tax_returns(tr, income, "X", 0.40, tax_exempt=True)
    at_taxed = st.after_tax_returns(tr, income, "X", 0.40, tax_exempt=False)
    # tax-exempt keeps the full 0.8%/mo; taxed loses 40% of the 0.3% coupon
    assert np.allclose(at_exempt.values, 0.008)
    assert np.allclose(at_taxed.values, 0.008 - 0.003 * 0.40)
    assert (at_exempt > at_taxed).all()


# --------------------------------------------------------------------------- #
# Performance / risk / cost helpers
# --------------------------------------------------------------------------- #
def test_costs_reduce_net():
    assert st.switch_cost_drag(30.0, 10.0) > st.switch_cost_drag(5.0, 10.0) > 0.0


def test_max_drawdown_on_known_path():
    px = pd.Series([100, 120, 60, 90], index=pd.bdate_range("2020-01-01", periods=4))
    dd = st.max_drawdown(px)
    assert abs(dd["depth_pct"] - (-50.0)) < 1e-9   # 120 -> 60 is -50%


def test_monthly_returns_drops_partial_month():
    # a tape ending mid-month must drop that final (partial) month
    idx = pd.bdate_range("2020-01-01", "2020-03-16")
    px = pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx)
    m = st.monthly_returns(px)
    assert m.index.max() <= pd.Timestamp("2020-02-29")


def test_era_table_and_calendar_year():
    idx = pd.period_range("2015-01", periods=48, freq="M").to_timestamp(how="end")
    rng = np.random.default_rng(3)
    m = pd.DataFrame({"HYD": rng.normal(0.006, 0.01, 48),
                      "MUB": rng.normal(0.004, 0.008, 48)}, index=idx)
    sp = st.premium_series(m)
    eras = st.era_table(sp, [("2015-01", "2016-12", "first"), ("2017-01", "2018-12", "second")])
    assert len(eras) == 2 and all("tstat" in e for e in eras)
    cy = st.calendar_year_table(m, ["HYD", "MUB"])
    assert "HYD-MUB" in cy.columns and len(cy) == 4
