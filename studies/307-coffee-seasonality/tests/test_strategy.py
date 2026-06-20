"""The frost-premium engine recovers the planted signal when it is real, the null shows nothing, the
block-bootstrap CI excludes 0 only when the signal is genuine, and the timer beats buy-and-hold only
when the signal is real. Offline, seeded — these tests never skip."""

import numpy as np
import pandas as pd

from coffee_seasonality import strategy as st


def test_frost_world_frost_above_harvest(frost_world):
    df, _ = frost_world
    r = st.frost_harvest_tstat(df["coffee"])
    assert r["frost_mean"] > r["harvest_mean"]
    assert r["tstat"] > 2.0


def test_null_world_no_seasonality(null_world):
    df, _ = null_world
    r = st.frost_harvest_tstat(df["coffee"])
    assert abs(r["tstat"]) < 2.0


def test_bootstrap_ci_excludes_zero_when_signal_real(frost_world):
    df, _ = frost_world
    ci = st.spread_bootstrap_ci(df["coffee"], n_boot=1000, seed=307)
    assert ci["lo"] > 0  # planted positive spread


def test_bootstrap_ci_straddles_zero_under_null(null_world):
    df, _ = null_world
    ci = st.spread_bootstrap_ci(df["coffee"], n_boot=1000, seed=307)
    assert ci["lo"] < 0 < ci["hi"]


def test_month_stats_shape_and_hac(frost_world):
    df, _ = frost_world
    ms = st.month_stats(df["coffee"])
    assert len(ms) == 12
    assert {"mean", "std", "n", "tstat", "tstat_hac"}.issubset(ms.columns)
    # HAC SE >= naive SE in general -> |t_HAC| typically <= |t_naive| under positive autocorr,
    # but both must be finite for every month.
    assert ms["tstat"].notna().all()
    assert ms["tstat_hac"].notna().all()


def test_timer_beats_buyhold_when_signal_real(frost_world):
    df, _ = frost_world
    timer = st.seasonal_timer(df["coffee"])
    bh = st.buy_hold(df["coffee"])
    assert st.summary(timer)["sharpe"] > st.summary(bh)["sharpe"]


def test_timer_in_frost_earns_coffee_return(frost_world):
    df, _ = frost_world
    timer = st.seasonal_timer(df["coffee"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    frost = timer[timer.index.month.isin(st.FROST_MONTHS)]
    coffee_fr = df["coffee"].reindex(frost.index)
    assert np.allclose(frost.values, coffee_fr.values)


def test_timer_in_harvest_earns_negative_coffee(frost_world):
    df, _ = frost_world
    timer = st.seasonal_timer(df["coffee"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    harvest = timer[timer.index.month.isin(st.HARVEST_MONTHS)]
    coffee_ha = df["coffee"].reindex(harvest.index)
    assert np.allclose(harvest.values, -coffee_ha.values)


def test_timer_flat_months_earn_tbill(frost_world):
    df, _ = frost_world
    tbill = pd.Series(0.003, index=df.index)
    timer = st.seasonal_timer(df["coffee"], tbill=tbill)
    timer.index = pd.DatetimeIndex(timer.index)
    flat_months = [m for m in range(1, 13) if m not in st.FROST_MONTHS + st.HARVEST_MONTHS]
    flat = timer[timer.index.month.isin(flat_months)]
    assert (flat == 0.003).all()


def test_summary_sharpe_convention(frost_world):
    df, _ = frost_world
    bh = st.buy_hold(df["coffee"])
    rf = pd.Series(0.002, index=df.index)
    raw = st.summary(bh)
    excess = st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert excess[k] == raw[k]


def test_apply_costs_reduces_return(frost_world):
    df, _ = frost_world
    timer = st.seasonal_timer(df["coffee"])
    net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10)
    assert (net <= timer + 1e-12).all()
    assert net.mean() < timer.mean()
