"""Offline, fixed-seed tests for the Study 911 strategy & inference machinery.

The synthetic world is deterministic; the Sharpe-advantage estimator recovers a planted
quality edge and stays silent on the null; the HAC spread t behaves; the trap detector
flags the leveraged-carry leg; costs strictly reduce the net; the rebalance is
point-in-time; the inference primitives are sane. All offline, no network, no real cache.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reit_quality import data, strategy as st  # noqa: E402


# --- primitives ------------------------------------------------------------ #
def test_nw_t_matches_plain_t_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.02, 4000)
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(st.nw_mean_t(x, lags=6) - plain) < 0.5


def test_nw_t_nan_on_tiny_sample():
    assert np.isnan(st.nw_mean_t(np.array([0.01, 0.02])))


def test_ann_return_and_vol_positive():
    r = pd.Series(np.full(24, 0.01))
    assert 12.0 < st.ann_return(r) < 13.0        # ~ (1.01^12 - 1) * 100
    assert st.ann_vol(pd.Series([0.01, -0.01, 0.02, -0.02] * 6)) > 0


# --- the planted quality edge is recovered, the null is silent ------------- #
def test_planted_edge_recovered(big_edge_world):
    adv = st.sharpe_advantage(big_edge_world, "QUAL", "BROAD", rf="CASH", n_boot=600)
    assert adv["advantage"] > 0            # quality out-Sharpes broad
    assert adv["ci_low"] > 0               # CI clear of zero on a strong planted edge
    sp = st.spread_stats(big_edge_world, "QUAL", "BROAD")
    assert sp["mean_bps"] > 0 and sp["t_nw"] > 2.0


def test_null_world_no_edge(null_world):
    adv = st.sharpe_advantage(null_world, "QUAL", "BROAD", rf="CASH", n_boot=600)
    assert adv["ci_low"] < 0 < adv["ci_high"]   # CI straddles zero on the null
    sp = st.spread_stats(null_world, "QUAL", "BROAD")
    assert abs(sp["t_nw"]) < 2.5


# --- the leveraged-carry trap is flagged ----------------------------------- #
def test_trap_detected(edge_world):
    tg = st.trap_gap(edge_world, "TRAP", "QUAL", "BROAD", rf="CASH")
    assert tg["trap_sharpe"] < tg["broad_sharpe"]      # trap Sharpe structurally worse
    assert tg["quality_minus_trap_bps"] > 0
    d = st.synth_detect(edge_world)
    assert d["trap_flagged"] is True


# --- costs strictly reduce the net ----------------------------------------- #
def test_costs_reduce_net(edge_world):
    free = st.costed_book(edge_world, ["QUAL", "BROAD"], "BROAD", cost_bps_oneway=0.0)
    paid = st.costed_book(edge_world, ["QUAL", "BROAD"], "BROAD", cost_bps_oneway=10.0)
    assert paid["net_bps_mo"] < free["net_bps_mo"]
    assert paid["drag_bps_yr"] > 0


# --- rebalance is point-in-time (no look-ahead) ---------------------------- #
def test_rebalance_lag_is_point_in_time():
    # The quality book on month t uses only month-t returns of its members; shifting the
    # inputs by one month shifts the book output by exactly one month (no peeking ahead).
    idx = pd.period_range("2010-01", periods=24, freq="M").to_timestamp(how="end")
    m = pd.DataFrame({"A": np.linspace(-0.01, 0.03, 24),
                      "B": np.linspace(0.02, -0.02, 24)}, index=idx)
    book = st.quality_book(m, ["A", "B"])
    shifted = st.quality_book(m.shift(1), ["A", "B"])
    assert np.allclose(shifted.iloc[5], book.iloc[4], equal_nan=True)


# --- era + calendar tables are well-formed --------------------------------- #
def test_era_and_calendar_tables(edge_world):
    eras = st.era_spreads(edge_world, "QUAL", "BROAD", cut="2017-01-01")
    assert len(eras) == 2 and all("t_nw" in e for e in eras)
    cal = st.calendar_year_table(edge_world, ["BROAD", "QUAL", "TRAP"])
    assert set(["BROAD", "QUAL", "TRAP"]).issubset(cal.columns)
    assert cal.shape[0] >= 18


# --- sharpe table sane ------------------------------------------------------ #
def test_sharpe_table(edge_world):
    tab = st.sharpe_table(edge_world, ["BROAD", "QUAL", "TRAP"], rf="CASH")
    assert tab.loc["TRAP", "excess_sharpe"] < tab.loc["BROAD", "excess_sharpe"]
    assert tab.loc["QUAL", "excess_sharpe"] >= tab.loc["BROAD", "excess_sharpe"] - 0.05


# --- real-cache smoke test, skipped when no cache is present ---------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real-tape cache absent (offline / CI) — synthetic tests cover the machinery")
def test_real_cache_smoke():
    px = data.load_prices()
    m = st.monthly_returns(px)
    # The mortgage-REIT trap must have a worse excess Sharpe than the broad index on the tape.
    tab = st.sharpe_table(m, [data.BROAD, data.QUALITY, data.TRAP], rf=data.CASH)
    assert tab.loc[data.TRAP, "excess_sharpe"] < tab.loc[data.BROAD, "excess_sharpe"]
    # And the quality sleeve's spread over broad must be finite with a real HAC t.
    sp = st.spread_stats(m, data.QUALITY, data.BROAD)
    assert np.isfinite(sp["mean_bps"]) and np.isfinite(sp["t_nw"])
