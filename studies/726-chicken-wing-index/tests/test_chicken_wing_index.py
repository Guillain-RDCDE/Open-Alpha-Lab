"""The Super-Bowl-window engine recovers a planted signal when it is real, shows nothing under the null,
the block-bootstrap CI excludes 0 only when the signal is genuine, the timer beats buy-and-hold only when
the signal is real, and the hardcoded wing-price proxy carries its cited supply-shock shape. Offline,
seeded — these tests never touch the network."""

import numpy as np
import pandas as pd

from chicken_wing_index import data, strategy as st


def test_superbowl_world_window_above_rest(superbowl_world):
    df, _ = superbowl_world
    r = st.superbowl_window_test(df["wing"])
    assert r["window_mean"] > r["rest_mean"]
    assert r["tstat"] > 2.0


def test_null_world_no_seasonality(null_world):
    df, _ = null_world
    r = st.superbowl_window_test(df["wing"])
    assert abs(r["tstat"]) < 2.0


def test_bootstrap_ci_excludes_zero_when_signal_real(superbowl_world):
    df, _ = superbowl_world
    ci = st.spread_bootstrap_ci(df["wing"], n_boot=1000, seed=726)
    assert ci["lo"] > 0  # planted positive window spread


def test_bootstrap_ci_straddles_zero_under_null(null_world):
    df, _ = null_world
    ci = st.spread_bootstrap_ci(df["wing"], n_boot=1000, seed=726)
    assert ci["lo"] < 0 < ci["hi"]


def test_month_stats_shape_and_hac(superbowl_world):
    df, _ = superbowl_world
    ms = st.month_stats(df["wing"])
    assert len(ms) == 12
    assert {"mean", "std", "n", "tstat", "tstat_hac"}.issubset(ms.columns)
    assert ms["tstat"].notna().all()
    assert ms["tstat_hac"].notna().all()


def test_placebo_ranks_january_top_when_signal_real(superbowl_world):
    df, _ = superbowl_world
    pl = st.placebo_months(df["wing"])
    assert pl.index[0] == 1  # planted premium is in January → January's timer ranks #1


def test_timer_beats_buyhold_when_signal_real(superbowl_world):
    df, _ = superbowl_world
    timer = st.superbowl_timer(df["wing"])
    bh = st.buy_hold(df["wing"])
    assert st.summary(timer)["sharpe"] > st.summary(bh)["sharpe"]


def test_timer_in_window_earns_wing_return(superbowl_world):
    df, _ = superbowl_world
    timer = st.superbowl_timer(df["wing"])
    timer.index = pd.DatetimeIndex(timer.index)
    df.index = pd.DatetimeIndex(df.index)
    win = timer[timer.index.month.isin(st.SUPERBOWL_MONTHS)]
    wing_win = df["wing"].reindex(win.index)
    assert np.allclose(win.values, wing_win.values)


def test_timer_flat_months_earn_tbill(superbowl_world):
    df, _ = superbowl_world
    tbill = pd.Series(0.003, index=df.index)
    timer = st.superbowl_timer(df["wing"], tbill=tbill)
    timer.index = pd.DatetimeIndex(timer.index)
    flat_months = [m for m in range(1, 13) if m not in st.SUPERBOWL_MONTHS]
    flat = timer[timer.index.month.isin(flat_months)]
    assert (flat == 0.003).all()


def test_window_alpha_recovers_positive_under_signal(superbowl_world):
    df, _ = superbowl_world
    al = st.window_alpha_vs_market(df["wing"], df["spy"])
    assert al["alpha_m"] > 0
    assert al["t_alpha"] > 2.0


def test_summary_sharpe_convention(superbowl_world):
    df, _ = superbowl_world
    bh = st.buy_hold(df["wing"])
    rf = pd.Series(0.002, index=df.index)
    raw = st.summary(bh)
    excess = st.summary(bh, rf=rf)
    assert excess["sharpe"] < raw["sharpe"]
    for k in ("cagr", "vol_ann", "max_drawdown"):
        assert excess[k] == raw[k]


def test_apply_costs_reduces_return(superbowl_world):
    df, _ = superbowl_world
    timer = st.superbowl_timer(df["wing"])
    net = st.apply_costs(timer, n_trades_per_year=2, cost_bps_one_way=10)
    assert (net <= timer + 1e-12).all()
    assert net.mean() < timer.mean()


def test_wing_price_proxy_supply_shock_shape():
    """The hardcoded, cited proxy must carry its load-bearing shape: a 2021 spike then a 2023 crash."""
    wp = data.load_wing_price()
    assert wp.loc["2021-01-31"] > 2.5           # record pandemic-shortage spike
    assert wp.loc["2023-01-31"] < 1.5           # post-shock collapse
    assert wp.loc["2021-01-31"] > wp.loc["2020-01-31"]  # spiked from the pre-2021 range
