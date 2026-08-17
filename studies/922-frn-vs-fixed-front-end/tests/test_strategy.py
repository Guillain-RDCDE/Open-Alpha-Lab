"""Race logic, lag discipline, inference and the study's spine — all offline/synthetic.

The spine: on a planted panel where the fixed leg has real duration, the regime estimator
must recover a large positive rising-minus-falling contrast; on a null panel with the same
rate cycle but a zero-duration fixed leg it must stay quiet. Costs and borrow only ever
reduce a difference; every ^IRX-derived object is lagged exactly one day.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frn_front import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Lag discipline (the study's single execution lag)
# --------------------------------------------------------------------------- #
def test_cash_leg_is_lagged_one_day():
    irx = pd.Series([1.0, 2.0, 3.0, 4.0],
                    index=pd.bdate_range("2020-01-01", periods=4))
    cash = st.cash_leg(irx, basis=252)
    assert np.isnan(cash.iloc[0])
    assert cash.iloc[1] == pytest.approx(1.0 / 100 / 252)
    assert cash.iloc[3] == pytest.approx(3.0 / 100 / 252)


def test_regime_label_is_lagged_and_uses_only_the_past():
    rng = np.random.default_rng(0)
    irx = pd.Series(np.cumsum(rng.normal(0, 0.05, 400)) + 3.0,
                    index=pd.bdate_range("2018-01-01", periods=400))
    lab = st.irx_regime(irx, window=63, thresh=0.25)
    irx2 = irx.copy()
    irx2.iloc[300:] += 5.0          # perturb only the future
    lab2 = st.irx_regime(irx2, window=63, thresh=0.25)
    assert (lab.iloc[:300].fillna("na") == lab2.iloc[:300].fillna("na")).all()
    assert lab.iloc[:63].isna().all()


def test_regime_labels_are_the_three_expected_values():
    prices, _ = data.synthetic_panel(seed=922)
    lab = st.irx_regime(prices["IRX"]).dropna()
    assert set(lab.unique()).issubset({"rising", "flat", "falling"})
    assert (lab == "rising").sum() > 100 and (lab == "falling").sum() > 100


# --------------------------------------------------------------------------- #
# Race mechanics
# --------------------------------------------------------------------------- #
def test_pair_race_is_antisymmetric(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    a = st.pair_race(rets, "frn", "fixed")
    b = st.pair_race(rets, "fixed", "frn")
    assert a["ann_diff_gross"] == pytest.approx(-b["ann_diff_gross"])
    assert a["tstat"] == pytest.approx(-b["tstat"])


def test_costs_and_borrow_only_reduce_the_difference(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    gross = st.pair_race(rets, "frn", "fixed")["ann_diff"]
    costed = st.pair_race(rets, "frn", "fixed", cost_bps=5.0)["ann_diff"]
    borrowed = st.pair_race(rets, "frn", "fixed", borrow_bps=50.0)["ann_diff"]
    assert costed < gross
    assert borrowed == pytest.approx(gross - 50.0 * 1e-4)


def test_borrow_sweep_is_monotone(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    sw = st.borrow_sweep(rets, "frn", "fixed")
    assert sw["net_ann_pct"].is_monotonic_decreasing


def test_cost_sweep_shrinks_with_a_longer_hold(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    sw = st.cost_sweep(rets, "frn", "fixed")
    one = sw[(sw["cost_bps"] == 5.0) & (sw["horizon_years"] == 1.0)]["friction_ann_pct"].iloc[0]
    three = sw[(sw["cost_bps"] == 5.0) & (sw["horizon_years"] == 3.0)]["friction_ann_pct"].iloc[0]
    assert one == pytest.approx(3.0 * three)


def test_excess_frame_cancels_in_a_difference(planted):
    """Cash cancels: the pairwise difference must be identical gross or excess-of-cash."""
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    cash = st.cash_leg(prices["IRX"])
    exc = st.excess_frame(rets, cash)
    d_raw = (rets["frn"] - rets["fixed"]).reindex(exc.index)
    d_exc = exc["frn"] - exc["fixed"]
    assert np.allclose(d_raw.dropna().to_numpy(), d_exc.dropna().to_numpy())


def test_cash_proxy_sweep_shifts_levels_not_pair_ranking(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    sw = st.cash_proxy_sweep(rets, prices["IRX"])
    assert set(sw.index) == {"IRX/252", "IRX/360"}
    # a 360 basis accrues less cash, so every excess return must be higher
    assert (sw.loc["IRX/360", ["ann_frn", "ann_bills", "ann_fixed"]].to_numpy()
            > sw.loc["IRX/252", ["ann_frn", "ann_bills", "ann_fixed"]].to_numpy()).all()


def test_drawdowns_are_non_positive(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    tbl = st.drawdown_table(rets, split="2018-01-01")
    assert (tbl["dd_full"] <= 0).all() and (tbl["dd_liquid"] <= 0).all()
    # the duration-carrying leg must draw down deeper than the floater
    assert tbl.loc["fixed", "dd_full"] < tbl.loc["frn", "dd_full"]


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_a_planted_slope():
    rng = np.random.default_rng(2)
    d = (rng.random(4000) < 0.3).astype(float)
    y = 0.5 + 2.0 * d + rng.normal(0, 1.0, 4000)
    X = np.column_stack([np.ones(4000), d])
    beta, se, t = st.hac_ols(y, X)
    assert beta[1] == pytest.approx(2.0, abs=0.15)
    assert t[1] > 5


def test_hac_ols_is_quiet_on_pure_noise():
    rng = np.random.default_rng(3)
    d = (rng.random(4000) < 0.3).astype(float)
    y = rng.normal(0, 1.0, 4000)
    X = np.column_stack([np.ones(4000), d])
    _, _, t = st.hac_ols(y, X)
    assert abs(t[1]) < 3


def test_welch_and_one_sample_finite(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    assert np.isfinite(st.one_sample_t(rets["frn"].to_numpy()))
    assert np.isfinite(st.welch_t(rets["frn"].to_numpy(), rets["fixed"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_ci_brackets_point(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    ci = st.block_bootstrap_ci(rets["frn"] - rets["fixed"], n_boot=400, seed=922)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_summary_fields(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    s = st.summary(rets["frn"])
    for k in ("ann_return", "cagr", "sharpe", "vol_ann", "max_drawdown", "tstat"):
        assert np.isfinite(s[k])


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_duration_produces_a_large_positive_contrast(planted):
    """With a real 1.85-year duration the rate direction must flip the ranking."""
    prices, _ = planted
    d = st.synthetic_detect(prices)
    assert d["contrast"] > 2.0
    assert d["contrast_t"] > 4.0
    assert d["rising_extra"] > 0 and d["falling_extra"] < 0


def test_null_no_spurious_contrast(null):
    """Same rate cycle, zero-duration fixed leg: nothing for the estimator to find."""
    prices, _ = null
    d = st.synthetic_detect(prices)
    assert abs(d["contrast"]) < 1.5
    # the regime classifier is just as busy — the quiet result is not an empty sample
    assert d["n_rising"] > 100 and d["n_falling"] > 100


def test_null_across_seeds_is_centred():
    cs = np.array([st.synthetic_detect(
        data.synthetic_panel(signal_strength=0.0, seed=922 + s)[0])["contrast"]
        for s in range(8)])
    ts = np.array([st.synthetic_detect(
        data.synthetic_panel(signal_strength=0.0, seed=922 + s)[0])["contrast_t"]
        for s in range(8)])
    assert abs(cs.mean()) < 0.5
    assert (np.abs(ts) >= 2).sum() <= 2


def test_contrast_scales_with_planted_duration():
    weak = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=922)[0])
    full = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=922)[0])
    assert full["contrast"] > weak["contrast"] > 0


def test_regime_param_sweep_keeps_the_sign(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    sw = st.regime_param_sweep(rets, prices["IRX"], "frn", "fixed",
                               windows=(42, 63), threshs=(0.10, 0.25))
    assert (sw["contrast"] > 0).all()


def test_regime_table_covers_the_three_regimes(planted):
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    reg = st.irx_regime(prices["IRX"])
    tbl = st.regime_table(rets, reg, pairs=[("frn", "fixed")])
    assert {"rising", "falling"}.issubset(set(tbl.index))
    assert tbl.loc["rising", "frn-fixed"] > tbl.loc["falling", "frn-fixed"]


def test_cycle_table_runs_on_the_declared_windows(planted):
    """The hardcoded Fed calendar is an assumption — it must at least be well formed."""
    prices, _ = planted
    rets = st.daily_returns(prices, cols=["frn", "bills", "fixed"])
    windows = [("first half", str(rets.index[0].date()), str(rets.index[len(rets) // 2].date())),
               ("second half", str(rets.index[len(rets) // 2].date()), str(rets.index[-1].date()))]
    tbl = st.cycle_table(rets, windows=windows)
    assert len(tbl) == 2
    assert (tbl["n_days"] > 100).all()
