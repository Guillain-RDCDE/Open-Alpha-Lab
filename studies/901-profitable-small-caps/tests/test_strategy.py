"""Offline, fixed-seed tests for the profitable-small-caps machinery.

The synthetic world is deterministic; a planted quality edge is recovered by the Sharpe race
and the HAC t on the daily return difference; the null shows nothing; the common-window slice
is leak-free; costs reduce the net; the inference primitives behave. All offline. A single
real-cache test is skipped when ``_cache/psc_prices.csv`` is absent (as on CI).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from profitable_small import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(edge=0.4, seed=901)
    assert np.allclose(edge_world["x_CALF"].to_numpy(), (w2["x_quality"]).to_numpy())


# --------------------------------------------------------------------------- #
# Planted edge recovered / null flat  (the machinery proof)
# --------------------------------------------------------------------------- #
def test_planted_edge_recovered(edge_world):
    p = st.pair_test(edge_world, "x_CALF", "x_IWM")
    assert p["sharpe_diff"] > 0.15          # quality clearly out-Sharpes plain
    assert p["t_nw_diff"] > 2.0             # HAC t on the daily difference clears the bar
    assert p["diff_ci_low"] > 0             # bootstrap Sharpe-diff CI clears zero
    assert p["mean_diff_bps"] > 0


def test_null_world_flat(null_world):
    p = st.pair_test(null_world, "x_CALF", "x_IWM")
    assert abs(p["t_nw_diff"]) < 2.0        # no daily-difference signal
    assert p["diff_ci_low"] < 0 < p["diff_ci_high"]   # CI straddles zero


def test_planted_edge_beats_null(edge_world, null_world):
    pe = st.pair_test(edge_world, "x_CALF", "x_IWM")
    pn = st.pair_test(null_world, "x_CALF", "x_IWM")
    assert pe["sharpe_diff"] > pn["sharpe_diff"] + 0.1


def test_beta_decomp_alpha_lifts_with_edge(edge_world, null_world):
    de = st.beta_decomp(edge_world, "x_CALF", ["x_IWM", "x_SPY"])
    dn = st.beta_decomp(null_world, "x_CALF", ["x_IWM", "x_SPY"])
    assert de["t_alpha"] > 2.0                       # planted alpha is detectable
    assert de["alpha_ann_pct"] > dn["alpha_ann_pct"] + 3.0
    assert 0.5 < de["betas"]["x_IWM"] < 1.5          # loads on the small-cap factor


# --------------------------------------------------------------------------- #
# Common-window slicing is leak-free
# --------------------------------------------------------------------------- #
def test_common_window_is_intersection():
    idx = pd.bdate_range("2020-01-01", periods=50)
    df = pd.DataFrame({"x_CALF": np.arange(50.0), "x_IWM": np.arange(50.0)}, index=idx)
    df.loc[df.index[:10], "x_CALF"] = np.nan       # young fund: first 10 days missing
    win = st.common_window(df, ["x_CALF", "x_IWM"])
    assert len(win) == 40
    assert win.index.min() == idx[10]

def test_trailing_nans_do_not_change_past_stats():
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 60)
    base = pd.DataFrame({"x_CALF": x, "x_IWM": x * 0.9}, index=idx)
    s_full = st.leg_stats(st.common_window(base, ["x_CALF", "x_IWM"])["x_CALF"].to_numpy())
    padded = base.copy()
    padded.loc[padded.index[40:], "x_IWM"] = np.nan   # partner goes missing later
    s_pad = st.leg_stats(st.common_window(padded, ["x_CALF", "x_IWM"])["x_CALF"].to_numpy())
    # The common window shrank to 40 rows; the first-40 CALF Sharpe is unchanged by what
    # happens after (no look-ahead / no future leakage into past stats).
    s_first40 = st.leg_stats(base["x_CALF"].to_numpy()[:40])
    assert np.isclose(s_pad["sharpe"], s_first40["sharpe"])


# --------------------------------------------------------------------------- #
# Costs reduce the net
# --------------------------------------------------------------------------- #
def test_costs_reduce_net_sharpe(edge_world):
    c = st.costed_race(edge_world, "x_CALF", "x_IWM", er_quality=0.59, er_plain=0.19)
    assert c["sharpe_q_net"] < c["sharpe_q_gross"]
    assert c["charge_ann_pct"] > 0


def test_isolation_costs_reduce_net(edge_world):
    it = st.isolation_trade(edge_world, "x_CALF", "x_IWM")
    assert it["net_ann_pct"] < it["gross_ann_pct"]
    assert it["charge_ann_pct"] > 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_annualized_sharpe_matches_manual():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0004, 0.01, 3000)
    manual = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert np.isclose(st.annualized_sharpe(x), manual)


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0005, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_sharpe_diff_ci_zero_for_identical():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0004, 0.01, 1500)
    df = pd.DataFrame({"a": x, "b": x}, index=pd.bdate_range("2015-01-01", periods=1500))
    d = st.sharpe_diff_ci(df["a"].to_numpy(), df["b"].to_numpy())
    assert np.isclose(d["diff"], 0.0)
    assert d["ci_low"] <= 0 <= d["ci_high"]


def test_max_drawdown_known():
    # +100% then -50% back to start: peak at 2.0, trough back to 1.0 => -50% drawdown.
    x = np.array([1.0, -0.5])
    assert np.isclose(st.max_drawdown(x), -0.5)


def test_max_drawdown_is_nonpositive(edge_world):
    dd = st.leg_stats(edge_world["x_CALF"].to_numpy())["maxdd_pct"]
    assert dd <= 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_race_common_window_shared(edge_world):
    r = st.race(edge_world, {"CALF": "x_CALF", "IWM": "x_IWM", "SPY": "x_SPY"})
    assert r["n"] > 2000
    assert set(r["legs"]) == {"CALF", "IWM", "SPY"}
    assert r["legs"]["CALF"]["sharpe"] > r["legs"]["IWM"]["sharpe"]  # planted edge


# --------------------------------------------------------------------------- #
# Real-cache smoke test (skipped when the cache is absent, e.g. on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="no real _cache present (offline/CI)")
def test_real_cache_race_runs():
    prices = data.load_prices()
    frame = data.daily_frame(prices, asof=data.AS_OF)
    legs = {"CALF": "x_CALF", "XSHQ": "x_XSHQ", "IWM": "x_IWM", "IJR": "x_IJR", "SPY": "x_SPY"}
    r = st.race(frame, legs)
    assert r["n"] > 1500
    for name in legs:
        assert np.isfinite(r["legs"][name]["sharpe"])
    # SPY beat every small-cap flavour on excess-Sharpe over this window (the real headline).
    assert r["legs"]["SPY"]["sharpe"] > r["legs"]["CALF"]["sharpe"]
