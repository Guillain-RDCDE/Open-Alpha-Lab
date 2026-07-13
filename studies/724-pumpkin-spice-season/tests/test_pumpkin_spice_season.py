"""The season engine recovers the planted PSL premium when it is real, the null shows nothing, the
block-bootstrap CI excludes 0 only when the signal is genuine, the placebo ranks the planted window
top, and the season-only timer beats the off-season only when the signal is real. Offline, seeded —
these tests never skip."""

import numpy as np
import pandas as pd

from pumpkin_spice_season import strategy as st


def test_season_world_above_off(psl_world):
    df, _ = psl_world
    r = st.season_tstat(df["excess"])
    assert r["season_mean"] > r["off_mean"]
    assert r["tstat"] > 2.0


def test_null_world_no_season(null_world):
    df, _ = null_world
    r = st.season_tstat(df["excess"])
    assert abs(r["tstat"]) < 2.0


def test_bootstrap_ci_excludes_zero_when_signal_real(psl_world):
    df, _ = psl_world
    ci = st.spread_bootstrap_ci(df["excess"], n_boot=1000, seed=724)
    assert ci["lo"] > 0  # planted positive spread


def test_bootstrap_ci_straddles_zero_under_null(null_world):
    df, _ = null_world
    ci = st.spread_bootstrap_ci(df["excess"], n_boot=1000, seed=724)
    assert ci["lo"] < 0 < ci["hi"]


def test_window_placebo_ranks_planted_window_top(psl_world):
    df, _ = psl_world
    wp = st.window_placebo(df["excess"])
    assert len(wp) == 12
    assert bool(wp.iloc[0]["is_psl"])  # the planted Aug-Nov window is the strongest


def test_month_stats_shape_and_hac(psl_world):
    df, _ = psl_world
    ms = st.month_stats(df["excess"])
    assert len(ms) == 12
    assert {"mean", "std", "n", "tstat", "tstat_hac"}.issubset(ms.columns)
    assert ms["tstat"].notna().all()
    assert ms["tstat_hac"].notna().all()


def test_spread_timer_beats_offseason_when_signal_real(psl_world):
    df, _ = psl_world
    timer = st.spread_timer(df["excess"])
    timer.index = pd.DatetimeIndex(timer.index)
    df2 = df.copy()
    df2.index = pd.DatetimeIndex(df2.index)
    in_season = timer.index.month.isin(st.SEASON_MONTHS)
    # in-season the timer earns the (positively-planted) excess; off-season it is flat (0)
    assert timer[in_season].mean() > 0
    assert np.allclose(timer[~in_season].values, 0.0)


def test_spread_timer_in_season_earns_excess(psl_world):
    df, _ = psl_world
    timer = st.spread_timer(df["excess"])
    timer.index = pd.DatetimeIndex(timer.index)
    df2 = df.copy()
    df2.index = pd.DatetimeIndex(df2.index)
    season = timer[timer.index.month.isin(st.SEASON_MONTHS)]
    ex = df2["excess"].reindex(season.index)
    assert np.allclose(season.values, ex.values)


def test_spread_timer_flat_months_earn_tbill(psl_world):
    df, _ = psl_world
    tbill = pd.Series(0.003, index=df.index)
    timer = st.spread_timer(df["excess"], tbill=tbill)
    timer.index = pd.DatetimeIndex(timer.index)
    flat = timer[~timer.index.month.isin(st.SEASON_MONTHS)]
    assert np.allclose(flat.values, 0.003)


def test_rotation_holds_sbux_in_season_spy_otherwise(psl_world):
    df, _ = psl_world
    rot = st.seasonal_rotation(df["sbux"], df["spy"])
    rot.index = pd.DatetimeIndex(rot.index)
    df2 = df.copy()
    df2.index = pd.DatetimeIndex(df2.index)
    in_season = rot.index.month.isin(st.SEASON_MONTHS)
    assert np.allclose(rot[in_season].values, df2["sbux"].reindex(rot.index)[in_season].values)
    assert np.allclose(rot[~in_season].values, df2["spy"].reindex(rot.index)[~in_season].values)


def test_summary_sharpe_convention(psl_world):
    df, _ = psl_world
    bh = st.buy_hold(df["sbux"])
    rf = pd.Series(0.002, index=df.index)
    raw = st.summary(bh)
    excess = st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert excess[k] == raw[k]


def test_apply_costs_reduces_return(psl_world):
    df, _ = psl_world
    rot = st.seasonal_rotation(df["sbux"], df["spy"])
    net = st.apply_costs(rot, n_trades_per_year=2, cost_bps_one_way=5)
    assert (net <= rot + 1e-12).all()
    assert net.mean() < rot.mean()
