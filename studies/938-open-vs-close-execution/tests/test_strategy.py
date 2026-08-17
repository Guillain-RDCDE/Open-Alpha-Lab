"""Rule logic, execution-arm invariants, and the study's spine — all offline/synthetic.

The spine: the two arms differ **only** on trade days, by ``Δweight × intraday``; on a tape
with planted intraday continuation the open fill wins with a clear HAC *t*; on the null the
gap is centred on zero and fires at roughly the nominal rate. Costs reduce both arms, the
opening-auction penalty only the open arm, and the one-day execution lag admits no
look-ahead.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_close_exec import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_period_end_mask_marks_last_day_of_period():
    idx = pd.bdate_range("2020-01-01", periods=120)
    m = st.period_end_mask(idx, freq="M")
    ends = idx[m.to_numpy()]
    # every marked date must be the last business day of its month within the window
    for d in ends[:-1]:
        same_month = idx[(idx.year == d.year) & (idx.month == d.month)]
        assert d == same_month[-1]
    mw = st.period_end_mask(idx, freq="W")
    assert mw.sum() > m.sum()  # more week ends than month ends


def test_rule_target_is_binary_and_warm_up_is_nan():
    frame, _ = data.synthetic_daily(seed=938)
    tgt = st.rule_target(frame["close"], freq="M", lookback=10)
    assert set(tgt.dropna().unique().tolist()).issubset({0.0, 1.0})
    assert tgt.isna().sum() > 100          # a 10-month warm-up
    assert tgt.iloc[0] != tgt.iloc[0] or np.isnan(tgt.iloc[0])


def test_rule_target_all_in_when_monotone_rising():
    close = pd.Series(np.linspace(100, 400, 600),
                      index=pd.bdate_range("2018-01-01", periods=600))
    tgt = st.rule_target(close, freq="M", lookback=10)
    assert tgt.dropna().eq(1.0).all()


def test_rule_target_all_out_when_monotone_falling():
    close = pd.Series(np.linspace(400, 100, 600),
                      index=pd.bdate_range("2018-01-01", periods=600))
    tgt = st.rule_target(close, freq="M", lookback=10)
    assert tgt.dropna().eq(0.0).all()


def test_no_lookahead_future_prices_do_not_move_past_targets():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))),
                      index=pd.bdate_range("2015-01-01", periods=900))
    t1 = st.rule_target(close, freq="M", lookback=10)
    close2 = close.copy()
    close2.iloc[700:] *= 4.0                       # perturb only the future
    t2 = st.rule_target(close2, freq="M", lookback=10)
    assert (t1.iloc[:700].fillna(-9) == t2.iloc[:700].fillna(-9)).all()


# --------------------------------------------------------------------------- #
# Execution-arm invariants
# --------------------------------------------------------------------------- #
def _arms(frame, **kw):
    return st.run_arms(frame[["open", "close"]], frame["cash"],
                       st.rule_target(frame["close"]), **kw)


def test_arms_differ_only_on_trade_days(null):
    frame, _ = null
    arms = _arms(frame, cost_bps=0.0)
    quiet = arms[arms["trade"] == 0]
    assert len(quiet) > 1000
    # Not bit-zero: the open arm compounds its two intra-day legs while the close arm holds
    # one, leaving a second-order cross-term of order (daily return)^2 ~ 1e-9. Nothing else.
    assert quiet["diff"].abs().max() < 1e-7


def test_gap_equals_delta_weight_times_intraday_excess(null):
    """The whole difference is the sliver: Δw x (asset intraday - cash intraday)."""
    frame, _ = null
    arms = _arms(frame, cost_bps=0.0, cash_overnight_frac=0.73)
    tr = arms[arms["trade"] > 0]
    dw = tr["w_id"] - tr["w_on"]
    sliver = dw * (tr["r_id"] - 0.27 * tr["r_cash"])
    # Exact up to the within-day compounding cross-term overnight x intraday, of order
    # (daily return)^2. The sliver explains essentially all of the venue difference.
    assert (tr["diff"] - sliver).abs().max() < 3e-3
    assert np.corrcoef(tr["diff"].to_numpy(), sliver.to_numpy())[0, 1] > 0.999


def test_execution_lag_is_exactly_one_day(null):
    """The open arm's intraday weight lags the target by 1; the close arm's by 2."""
    frame, _ = null
    arms = _arms(frame)
    tgt = st.rule_target(frame["close"]).reindex(arms.index)
    assert (arms["w_id"].to_numpy()[2:] == tgt.shift(1).to_numpy()[2:]).all()
    assert (arms["w_cc"].to_numpy()[2:] == tgt.shift(2).to_numpy()[2:]).all()


def test_costs_monotonically_reduce_both_arms(null):
    frame, _ = null
    means = []
    for c in (0.0, 2.0, 10.0):
        arms = _arms(frame, cost_bps=c)
        means.append((arms["r_open"].mean(), arms["r_close"].mean()))
    assert means[0][0] >= means[1][0] >= means[2][0]
    assert means[0][1] >= means[1][1] >= means[2][1]


def test_opening_penalty_only_hurts_the_open_arm(null):
    frame, _ = null
    a0 = _arms(frame, cost_bps=2.0, open_extra_bps=0.0)
    a5 = _arms(frame, cost_bps=2.0, open_extra_bps=5.0)
    assert np.allclose(a0["r_close"].to_numpy(), a5["r_close"].to_numpy())
    assert a5["r_open"].mean() < a0["r_open"].mean()
    assert st.venue_gap(a5)["gap_ann_bps"] < st.venue_gap(a0)["gap_ann_bps"]


def test_cash_split_proxy_barely_moves_the_gap(null):
    frame, _ = null
    gaps = [st.venue_gap(_arms(frame, cash_overnight_frac=f))["gap_ann_bps"]
            for f in (0.0, 0.5, 1.0)]
    assert max(gaps) - min(gaps) < 2.0  # bps/yr — the PROXY cannot drive the answer


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_finite(null):
    frame, _ = null
    arms = _arms(frame)
    assert np.isfinite(st.one_sample_t(arms["diff"].to_numpy()))
    assert np.isfinite(st.welch_t(arms["e_open"].to_numpy(), arms["e_close"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_cluster_t_is_more_conservative_when_fills_share_a_date():
    """Duplicate every fill onto its own date: the naive t inflates, the clustered t must not."""
    rng = np.random.default_rng(938)
    dates = pd.bdate_range("2010-01-01", periods=200)
    base = rng.normal(0.0004, 0.01, 200)
    # four perfectly correlated "tapes" print the same shock on the same day
    x = np.repeat(base, 4)
    d = np.repeat(dates.to_numpy(), 4)
    naive = st.one_sample_t(x)
    clustered = st.cluster_t(x, d)
    assert abs(clustered) < abs(naive)                      # the correction bites
    assert abs(clustered - st.one_sample_t(base)) < 0.05     # ~the honest one-per-day t


def test_cluster_bootstrap_is_deterministic_and_wider_than_wilson():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2012-01-01", periods=150)
    base = rng.normal(0.0002, 0.01, 150)
    x, d = np.repeat(base, 4), np.repeat(dates.to_numpy(), 4)
    a = st.cluster_bootstrap_trades(x, d, n_boot=300, seed=938)
    b = st.cluster_bootstrap_trades(x, d, n_boot=300, seed=938)
    assert a["mean_ci"] == b["mean_ci"] and a["win_ci"] == b["win_ci"]
    assert a["n_fills"] == 600 and a["n_dates"] == 150
    k = int((x > 0).sum())
    wlo, whi = st.wilson_interval(k, len(x))
    assert (a["win_ci"][1] - a["win_ci"][0]) > (whi - wlo)   # independence was a lie


def test_block_bootstrap_ci_brackets_point(null):
    frame, _ = null
    arms = _arms(frame)
    ci = st.block_bootstrap_ci(arms["diff"], n_boot=400, seed=938)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    cis = st.block_bootstrap_ci(arms["e_close"], stat="sharpe", n_boot=400, seed=938)
    assert cis["ci_low"] <= cis["point"] <= cis["ci_high"]


# --------------------------------------------------------------------------- #
# The study's spine — the detector recovers a planted effect and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_continuation_favours_the_open_fill(planted):
    frame, _ = planted
    d = st.synthetic_detect(frame)
    assert d["gap_ann_bps"] > 40.0
    assert d["gap_t_hac"] > 2.0


def test_planted_effect_recovered_across_seeds():
    ts = np.array([
        st.synthetic_detect(data.synthetic_daily(signal_strength=1.0, seed=938 + 11 * s)[0])["gap_t_hac"]
        for s in range(6)
    ])
    assert (ts > 2.0).sum() >= 5


def test_null_is_quiet(null):
    frame, _ = null
    d = st.synthetic_detect(frame)
    assert abs(d["gap_t_hac"]) < 2.0


def test_null_across_seeds_is_centred():
    ts = np.array([
        st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=938 + 11 * s)[0])["gap_t_hac"]
        for s in range(8)
    ])
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 2


def test_close_arm_is_invariant_to_the_planted_split(planted, null):
    """The knob cannot touch the close-executed arm — the null is airtight by construction."""
    f1, _ = planted
    f0, _ = null
    a1, a0 = _arms(f1), _arms(f0)
    assert np.allclose(a1["r_close"].to_numpy(), a0["r_close"].to_numpy())


# --------------------------------------------------------------------------- #
# Panel, eras, sweeps
# --------------------------------------------------------------------------- #
def test_run_panel_pools_four_tapes(panel_null):
    frames, _ = panel_null
    cash = frames["SYN0"]["cash"]
    res = st.run_panel({k: v[["open", "close"]] for k, v in frames.items()}, cash)
    assert len(res["tickers"]) == 4
    g = st.venue_gap(res["pooled"])
    assert np.isfinite(g["gap_ann_bps"]) and abs(g["gap_t_hac"]) < 3.0


def test_era_cut_returns_both_halves(null):
    frame, _ = null
    arms = _arms(frame)
    mid = str(arms.index[len(arms) // 2].date())
    eras = st.era_cut(arms, split=mid)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gap_ann_bps"])


def test_cost_sweep_is_monotone_decreasing(null):
    frame, _ = null
    rows = st.cost_sweep(frame[["open", "close"]], frame["cash"], extra_grid=(0.0, 2.0, 10.0))
    gaps = [r["gap_ann_bps"] for r in rows]
    assert gaps[0] > gaps[1] > gaps[2]


def test_trade_day_decomposition_counts_add_up(null):
    frame, _ = null
    arms = _arms(frame)
    d = st.trade_day_decomposition(arms)
    assert d["n_entries"] + d["n_exits"] == int((arms["trade"] > 0).sum())
    assert abs(d["n_entries"] - d["n_exits"]) <= 1


def test_summary_fields_present(null):
    frame, _ = null
    arms = _arms(frame)
    s = st.summary(arms["e_close"])
    assert {"cagr", "sharpe", "vol_ann", "max_drawdown", "tstat"}.issubset(s)
    assert s["max_drawdown"] <= 0.0
