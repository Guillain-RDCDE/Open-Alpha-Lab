"""Offline, fixed-seed tests for the preferred-reset machinery.

The synthetic world is deterministic; the planted regime-contingent reset premium is
recovered (positive spread, HAC t lights up, concentrated in the high-rate regime); the null
stays silent; the regime-switch is point-in-time (one lag, no look-ahead); costs reduce the
net; the inference primitives behave. All offline, synthetic-only — the real-cache test is
skipped when _cache/ is absent (as on CI)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pref_reset import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism & synthetic world shape
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    w2 = data.synthetic_world(edge=0.0030, dur_hit=0.010, seed=909)
    for c in ("variable", "fixed", "cash", "regime"):
        assert np.allclose(planted_world[c].to_numpy(), w2[c].to_numpy())


def test_world_index_is_period(planted_world):
    # PeriodIndex kept (no ns-Timestamp overflow trap).
    assert isinstance(planted_world.index, pd.PeriodIndex)
    assert planted_world["regime"].isin([0.0, 1.0]).all()
    assert 0.0 < planted_world["regime"].mean() < 1.0


# --------------------------------------------------------------------------- #
# The planted edge is recovered; the null is silent
# --------------------------------------------------------------------------- #
def test_planted_edge_recovered(planted_world):
    d = st.synthetic_detect(planted_world)
    assert d["t_nw"] > 2.5                       # spread lights up
    assert d["spread_ann_pct"] > 0
    assert d["sharpe_adv"] > 0                    # variable out-carries fixed excess-of-cash


def test_planted_edge_is_regime_contingent(planted_world):
    d = st.synthetic_detect(planted_world)
    # the premium lives in the high-rate regime, ~nothing in the low-rate regime
    assert d["spread_high_ann_pct"] > d["spread_low_ann_pct"] + 5.0


def test_null_world_no_signal(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["t_nw"]) < 2.5
    assert abs(d["sharpe_adv"]) < 0.5


def test_null_rarely_fires_across_seeds():
    fires = 0
    for s in range(20):
        w = data.synthetic_world(edge=0.0, dur_hit=0.0, seed=909 + s)
        if abs(st.synthetic_detect(w)["t_nw"]) >= 2.0:
            fires += 1
    assert fires <= 2                             # ~5% false-positive rate


# --------------------------------------------------------------------------- #
# No look-ahead in the regime-switch; costs reduce the net
# --------------------------------------------------------------------------- #
def test_rate_signal_is_lagged():
    # A hand-built sleeve frame: the switch signal is the PRIOR month-end value (one shift).
    idx = pd.period_range("2015-01", periods=24, freq="M")
    sl = pd.DataFrame({
        "variable": np.linspace(0.01, 0.02, 24),
        "fixed": np.linspace(0.02, 0.01, 24),
        "cash": np.full(24, 0.001) + np.linspace(0.0, 0.004, 24),
    }, index=idx)
    raw = st.rate_signal(sl, lookback=6)
    # switch_strategy consumes rate_signal(...).shift(1); verify the shift is present by
    # confirming the strategy uses n = len - lookback - 1 rows (one dropped for the shift).
    sw = st.switch_strategy(sl, lookback=6)
    assert sw["n"] == 24 - 6 - 1                  # 6 for the lookback NaNs, 1 for the lag shift
    assert not np.isnan(raw.iloc[-1])


def test_costs_reduce_net():
    idx = pd.period_range("2015-01", periods=120, freq="M")
    rng = np.random.default_rng(0)
    rf = pd.DataFrame({
        "var_ex": rng.normal(0.002, 0.02, 120),
        "fix_ex": rng.normal(0.000, 0.02, 120),
        "spread": rng.normal(0.002, 0.01, 120),
    }, index=idx)
    gross = st.costed_spread(rf, cost_bps_oneway=0.0, borrow_annual_bps=0.0)["net_ann_pct"]
    net = st.costed_spread(rf, cost_bps_oneway=8.0, borrow_annual_bps=40.0)["net_ann_pct"]
    assert net < gross


def test_switch_never_beats_free_lunch_magnitude(planted_world):
    # sanity: switch strategy returns finite, plausible numbers on a synthetic sleeve frame
    sl = planted_world.rename(columns={})[["variable", "fixed", "cash"]]
    sw = st.switch_strategy(sl, lookback=6)
    assert 0.0 <= sw["share_variable"] <= 1.0
    assert np.isfinite(sw["switch_ex_sharpe"])
    assert sw["switches"] >= 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_ann_sharpe_sign_and_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.02, 600)              # strongly positive mean
    assert st.ann_sharpe(x) > 0
    assert st.ann_sharpe(-x) < 0


def test_max_drawdown_bounds():
    x = np.array([0.1, -0.5, 0.1, 0.1])
    dd = st.max_drawdown(x)
    assert -1.0 < dd < 0.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(30, 54)
    assert lo < 30 / 54 < hi


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    x = rng.normal(0.003, 0.01, 200)
    pt, lo, hi = st.block_bootstrap_ci(x, lambda a: float(np.mean(a)), block=6, n_boot=500)
    assert lo <= pt <= hi


# --------------------------------------------------------------------------- #
# Race frame / era cut on synthetic sleeves
# --------------------------------------------------------------------------- #
def test_race_and_era_on_synthetic(planted_world):
    sl = planted_world[["variable", "fixed", "cash"]]
    rf = st.race_frame(sl)
    h = st.race_stats(rf)
    assert h["n"] == len(planted_world)
    assert h["sharpe_adv"] > 0
    # era cut splits by the PeriodIndex ordering; both halves are non-empty
    ts = pd.Timestamp(planted_world.index[len(planted_world) // 2].to_timestamp())
    era = st.era_cut(rf, str(ts.date()))
    assert era["low_rate"]["n"] > 0 and era["high_rate"]["n"] > 0
