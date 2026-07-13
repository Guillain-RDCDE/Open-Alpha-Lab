"""The guac-seasonal engine recovers a planted premium, the null shows nothing, the placebo puts a
real signal in the extreme tail, and the timer beats buy-and-hold only when the signal is real.
Offline, seeded — these tests never touch the network."""

import numpy as np
import pandas as pd

from guacamole_bowl import data, strategy as st


def test_avocado_seasonal_is_soft_in_winter():
    """The premise check: the hardcoded wholesale-Hass window sits BELOW the annual mean."""
    av = data.avocado_window_vs_year()
    assert av["gap"] < 0  # winter is soft, undercutting the 'surge' before the tape
    assert av["window_mean"] < av["year_mean"]


def test_guac_world_window_above_rest(guac_world):
    df, _ = guac_world
    r = st.window_spread_tstat(df["pep"])
    assert r["window_mean"] > r["rest_mean"]
    assert r["tstat"] > 2.0


def test_null_world_no_seasonality(null_world):
    df, _ = null_world
    r = st.window_spread_tstat(df["pep"])
    assert abs(r["tstat"]) < 2.0


def test_placebo_ranks_real_signal_in_tail(guac_world):
    df, _ = guac_world
    pb = st.placebo_pairs(df["pep"])
    assert pb["n_pairs"] == 66
    assert pb["rank"] >= 60  # a real Jan-Feb premium sits in the extreme upper tail of 66 pairs


def test_placebo_null_is_ordinary(null_world):
    df, _ = null_world
    pb = st.placebo_pairs(df["pep"])
    assert 5 <= pb["rank"] <= 62  # under the null Jan-Feb is unremarkable, not an outlier


def test_bootstrap_ci_excludes_zero_when_signal_real(guac_world):
    df, _ = guac_world
    ci = st.spread_bootstrap_ci(df["pep"], n_boot=1000, seed=723)
    assert ci["lo"] > 0  # planted positive spread


def test_bootstrap_ci_straddles_zero_under_null(null_world):
    df, _ = null_world
    ci = st.spread_bootstrap_ci(df["pep"], n_boot=1000, seed=723)
    assert ci["lo"] < 0 < ci["hi"]


def test_month_stats_shape_and_hac(guac_world):
    df, _ = guac_world
    ms = st.month_stats(df["pep"])
    assert len(ms) == 12
    assert {"mean", "std", "n", "tstat", "tstat_hac"}.issubset(ms.columns)
    assert ms["tstat"].notna().all()
    assert ms["tstat_hac"].notna().all()


def test_timer_beats_buyhold_when_signal_real(guac_world):
    df, _ = guac_world
    timer = st.seasonal_timer(df["pep"], tbill=df["tbill"])
    bh = st.buy_hold(df["pep"])
    assert st.summary(timer, rf=df["tbill"])["sharpe"] > st.summary(bh, rf=df["tbill"])["sharpe"]


def test_timer_in_window_earns_asset_return(guac_world):
    df, _ = guac_world
    timer = st.seasonal_timer(df["pep"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    win = timer[timer.index.month.isin(st.GUAC_MONTHS)]
    asset = df["pep"].reindex(win.index)
    assert np.allclose(win.values, asset.values)


def test_timer_flat_months_earn_tbill(guac_world):
    df, _ = guac_world
    tbill = pd.Series(0.003, index=df.index)
    timer = st.seasonal_timer(df["pep"], tbill=tbill)
    timer.index = pd.DatetimeIndex(timer.index)
    flat = [m for m in range(1, 13) if m not in st.GUAC_MONTHS]
    got = timer[timer.index.month.isin(flat)]
    assert (got == 0.003).all()


def test_summary_sharpe_convention(guac_world):
    df, _ = guac_world
    bh = st.buy_hold(df["pep"])
    rf = pd.Series(0.002, index=df.index)
    raw = st.summary(bh)
    excess = st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert excess[k] == raw[k]


def test_apply_costs_reduces_return(guac_world):
    df, _ = guac_world
    timer = st.seasonal_timer(df["pep"])
    net = st.apply_costs(timer, n_trades_per_year=2, cost_bps_one_way=5)
    assert (net <= timer + 1e-12).all()
    assert net.mean() < timer.mean()


def test_newey_west_alpha_recovers_beta(guac_world):
    df, _ = guac_world
    # regress the asset on itself-plus-noise proxy: beta ~ 1 sanity on a clean construction
    nw = st.newey_west_alpha_t(df["pep"], df["pep"], lags=6)
    assert abs(nw["beta"] - 1.0) < 1e-6
    assert abs(nw["alpha_m"]) < 1e-6
