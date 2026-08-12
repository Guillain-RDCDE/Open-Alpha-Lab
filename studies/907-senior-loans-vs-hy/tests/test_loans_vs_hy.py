"""Offline, fixed-seed tests for the senior-loans-vs-HY machinery.

The synthetic world is deterministic; the loan leg carries lower volatility by construction;
a planted risk-adjusted edge is recovered (loans win the bootstrap Sharpe race) while the
null — lower vol but equal Sharpe — shows no advantage; the costed long-short reduces the
net; the stress table sees the planted seniority cushion; the rebalance lag is
no-look-ahead; the inference primitives behave. All offline, no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from loans_vs_hy import data, strategy as st  # noqa: E402


def test_world_deterministic():
    a, _ = data.synthetic_pair(sharpe_edge=0.35, seed=907, n_days=1200)
    b, _ = data.synthetic_pair(sharpe_edge=0.35, seed=907, n_days=1200)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_loan_leg_has_lower_vol(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    assert ret["LOANS"].std() < ret["HY"].std()  # seniority + floating rate => lower vol


def _ensemble_advantage(edge, seeds):
    """Point excess-Sharpe advantage (loans − HY) averaged over a fixed seed set.

    The single-sample Sharpe-difference is inherently noisy (~0.24 SD over ~16y — the very
    reason the REAL advantage's CI straddles zero), so the machinery proof reads the
    ensemble mean over fixed seeds: deterministic and robust.
    """
    advs = []
    for s in seeds:
        frame, _ = data.synthetic_pair(sharpe_edge=edge, seed=s, n_days=4000)
        ret = st.to_returns(frame)
        advs.append(st.sharpe_advantage(
            st.excess(ret["LOANS"], ret["CASH"]),
            st.excess(ret["HY"], ret["CASH"]))["advantage"])
    return np.asarray(advs)


def test_planted_edge_recovered():
    advs = _ensemble_advantage(0.6, range(907, 917))
    assert advs.mean() > 0.30            # loans' excess-Sharpe clearly exceeds HY's
    assert (advs > 0).sum() >= 8         # positive in almost every seed


def test_planted_edge_bootstrap(edge_world):
    frame, _ = edge_world
    det = st.synthetic_detect(frame, n_boot=1500, seed=907)
    assert det["advantage"] > 0.10
    assert det["frac_loans_wins"] > 0.80       # bootstrap keeps the advantage above zero
    assert det["ci95"][1] > 0.0                # CI upper end positive


def test_null_world_no_advantage():
    advs = _ensemble_advantage(0.0, range(907, 917))
    # lower vol is exactly offset by lower carry -> Sharpes equal, advantage within noise
    assert abs(advs.mean()) < 0.15
    assert 2 <= (advs > 0).sum() <= 8          # sign scatters, no systematic advantage


def test_null_spread_not_significant(null_world):
    frame, _ = null_world
    sp = st.spread_stats(st.to_returns(frame)["LOANS"], st.to_returns(frame)["HY"])
    assert abs(sp["t_nw"]) < 2.5


def test_costs_reduce_net(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    gross = st.costed_long_short(ret["LOANS"], ret["HY"], cost_bps=0.0, borrow_bps_yr=0.0)
    net = st.costed_long_short(ret["LOANS"], ret["HY"], cost_bps=5.0, borrow_bps_yr=60.0)
    assert net["net_ann_pct"] < gross["net_ann_pct"]
    assert net["cost_bps_per_day"] > 0.0


def test_stress_table_sees_seniority_cushion(edge_world):
    frame, truth = edge_world
    lo = pd.Timestamp(truth["stress_peak"]).strftime("%Y-%m-%d")
    hi = pd.Timestamp(truth["stress_trough"]).strftime("%Y-%m-%d")
    rows = st.stress_table(frame, [(lo, hi, "planted stress")])
    r = rows[0]
    assert r["LOANS"] > r["HY"]        # loans fall less than HY over the planted window
    assert r["HY"] < 0.0


def test_rebalance_lag_no_lookahead():
    ret = pd.DataFrame(
        {"L": np.linspace(-0.01, 0.03, 40), "H": np.linspace(0.0, 0.01, 40)},
        index=pd.bdate_range("2020-01-01", periods=40),
    )
    a = st.costed_long_short(ret["L"], ret["H"], lag=1)
    b = st.costed_long_short(ret["L"], ret["H"], lag=0)
    # the lag shifts the traded spread by one day -> the two nets differ
    assert a["net_ann_pct"] != b["net_ann_pct"]


def test_composite_skips_nan():
    ret = pd.DataFrame(
        {"BKLN": [0.01, 0.02, 0.03], "SRLN": [np.nan, 0.04, 0.06]},
        index=pd.bdate_range("2013-01-01", periods=3),
    )
    comp = st.composite(ret, ("BKLN", "SRLN"))
    assert comp.iloc[0] == pytest.approx(0.01)          # BKLN alone before SRLN lists
    assert comp.iloc[1] == pytest.approx(0.03)          # mean once both trade


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    rng = np.random.default_rng(1)
    a = rng.normal(0.002, 0.01, 2000)
    b = rng.normal(0.000, 0.01, 2000)
    assert st.welch_t(a, b) > 2.0


def test_arm_stats_sane(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    s = st.arm_stats(ret["LOANS"], ret["CASH"])
    assert -1.0 < s["max_dd"] < 0.0
    assert s["vol"] > 0.0
    assert np.isfinite(s["sharpe"])


# ---- real-cache test: only runs where the tape has been fetched (skipped on CI) ----
@pytest.mark.skipif(not data.have_real(), reason="real tape cache absent (offline/CI)")
def test_real_cache_loads_and_windows():
    px = data.load_prices()
    assert px.index.max() <= pd.Timestamp(data.AS_OF)
    assert "BKLN" in px.columns and "HYG" in px.columns and "BIL" in px.columns
    # BKLN bounds the flagship window
    bk = px["BKLN"].dropna()
    assert bk.index.min() >= pd.Timestamp("2011-03-01")
    fp = data.fingerprint(px)
    assert isinstance(fp, str) and len(fp) == 12
