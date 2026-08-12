"""Offline, fixed-seed tests for the Quality-Income machinery.

The synthetic world is deterministic; the planted quality-over-yield edge is recovered
(positive Sharpe gap, HAC *t* on the monthly difference > 2); the null shows nothing;
sleeves are equal-weight monthly-rebalanced means; the turnover/cost lag uses no future
information; costs reduce the net; drawdown / calendar / inference primitives behave.
All offline (synthetic-only) — nothing here needs the real _cache/.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from quality_income import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, planted edge, null
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(n_months=150, edge=0.03, seed=900)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_world_index_is_period_not_timestamp(edge_world):
    # OOB-safe: the synthetic index must stay a PeriodIndex (never .to_timestamp on a
    # large monthly span, which overflows the pandas ns horizon on CI).
    assert isinstance(edge_world.index, pd.PeriodIndex)


def test_planted_edge_recovered(edge_world):
    d = st.synthetic_detect(edge_world, lags=6)
    assert d["sharpe_gap"] > 0.0        # quality beats yield on Sharpe
    assert d["t_nw"] > 2.0              # the monthly difference lights up
    assert d["diff_ann_pct"] > 0.0


def test_null_world_no_signal(null_world):
    d = st.synthetic_detect(null_world, lags=6)
    assert abs(d["t_nw"]) < 2.5         # quiet on the null


def test_null_gap_small(null_world):
    d = st.synthetic_detect(null_world, lags=6)
    assert abs(d["sharpe_gap"]) < 0.35  # no material Sharpe advantage at edge=0


# --------------------------------------------------------------------------- #
# Sleeve construction
# --------------------------------------------------------------------------- #
def test_sleeve_is_equal_weight_mean():
    m = pd.DataFrame({"A": [0.10, -0.05, 0.02], "B": [0.00, 0.05, 0.04]},
                     index=pd.period_range("2020-01", periods=3, freq="M"))
    s = st.sleeve_returns(m, ["A", "B"])
    assert np.allclose(s.to_numpy(), [0.05, 0.00, 0.03])


def test_sleeve_drops_rows_until_all_members_listed():
    # B lists one month late -> the sleeve must not start on a partial basket.
    m = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [np.nan, 0.04, 0.05]},
                     index=pd.period_range("2020-01", periods=3, freq="M"))
    s = st.sleeve_returns(m, ["A", "B"])
    assert len(s) == 2
    assert s.index[0] == pd.Period("2020-02", freq="M")


# --------------------------------------------------------------------------- #
# Costs / turnover — no look-ahead, and they bite
# --------------------------------------------------------------------------- #
def test_turnover_nonneg_and_no_lookahead():
    m = pd.DataFrame({"A": [0.10, -0.08, 0.05, 0.03], "B": [-0.02, 0.06, -0.04, 0.01]},
                     index=pd.period_range("2020-01", periods=4, freq="M"))
    t = st.sleeve_turnover(m, ["A", "B"])
    assert (t.to_numpy() >= 0).all()
    # turnover on month t uses only within-month drift of A,B at t (no future rows):
    # truncating the frame after t leaves earlier turnover values unchanged.
    t_short = st.sleeve_turnover(m.iloc[:3], ["A", "B"])
    assert np.allclose(t.to_numpy()[:3], t_short.to_numpy())


def test_costs_reduce_net(edge_world):
    # Build a tiny two-ETF frame from the synthetic sleeves; costs must lower net Sharpe.
    m = pd.DataFrame({"A": edge_world["quality"].to_numpy(),
                      "B": edge_world["quality"].to_numpy() * 0.9},
                     index=edge_world.index)
    cash = pd.Series(0.0, index=edge_world.index, name="cash")
    lo = st.costed_sleeve(m, ["A", "B"], cash, one_way_bps=0.0)
    hi = st.costed_sleeve(m, ["A", "B"], cash, one_way_bps=50.0)
    assert hi["net_cagr"] < lo["net_cagr"]
    assert hi["cost_drag_bps_yr"] > lo["cost_drag_bps_yr"]


# --------------------------------------------------------------------------- #
# Excess-of-cash: the gap is cash-independent
# --------------------------------------------------------------------------- #
def test_sharpe_gap_cash_independent(edge_world):
    q = edge_world["quality"].rename("q")
    y = edge_world["yield"].rename("y")
    zero = pd.Series(0.0, index=edge_world.index, name="cash")
    const = pd.Series(0.002, index=edge_world.index, name="cash")
    d0 = st.sharpe_gap_test(q, y, zero, lags=6)
    dc = st.sharpe_gap_test(q, y, const, lags=6)
    # The quality-minus-yield spread (and its HAC t) cancels cash exactly.
    assert abs(d0["t_nw"] - dc["t_nw"]) < 1e-9
    assert abs(d0["diff_mean_bps"] - dc["diff_mean_bps"]) < 1e-9


# --------------------------------------------------------------------------- #
# Performance / risk helpers
# --------------------------------------------------------------------------- #
def test_max_drawdown_known():
    # +50% then -50% -> wealth 1.5 then 0.75 -> drawdown -50%.
    r = pd.Series([0.5, -0.5], index=pd.period_range("2020-01", periods=2, freq="M"))
    assert abs(st.max_drawdown(r) - (-0.5)) < 1e-12


def test_calendar_year_table_shape():
    idx = pd.period_range("2019-01", periods=24, freq="M")
    q = pd.Series(0.01, index=idx); y = pd.Series(0.00, index=idx)
    tab = st.calendar_year_table({"Q": q, "Y": y})
    assert list(tab.columns) == ["Q", "Y"]
    assert set(tab.index) == {2019, 2020}
    assert tab.loc[2019, "Q"] == pytest.approx((1.01 ** 12) - 1.0)


def test_era_cut_returns_both_halves(edge_world):
    q = edge_world["quality"].rename("q"); y = edge_world["yield"].rename("y")
    cash = pd.Series(0.0, index=edge_world.index, name="cash")
    # era_cut compares Timestamps; give it a Timestamp index for this check.
    ts = q.copy(); ts.index = edge_world.index.to_timestamp(how="end")
    ys = y.copy(); ys.index = ts.index
    cs = cash.copy(); cs.index = ts.index
    eras = st.era_cut(ts, ys, cs, split="2020-01-01", lags=6)
    assert "early" in eras and "late" in eras


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.02, 3000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_detects_mean_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 0.01, 500)
    b = rng.normal(0.00, 0.01, 500)
    assert st.welch_t(a, b) > 5


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_bootstrap_gap_positive_on_planted(edge_world):
    q = edge_world["quality"].rename("q"); y = edge_world["yield"].rename("y")
    cash = pd.Series(0.0, index=edge_world.index, name="cash")
    bs = st.sharpe_gap_bootstrap(q, y, cash, n_draws=800, seed=900)
    assert bs["obs"] > 0
    assert bs["frac_negative"] < 0.2    # planted gap keeps the CI mostly clear of zero


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped when the git-ignored _cache/ is absent (CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.TAPE_CACHE),
                    reason="real yfinance cache absent (offline / CI)")
def test_real_cache_loads_and_sleeves_build():
    prices = data.load_prices()
    assert set(data.TICKERS).issubset(set(prices.columns))
    mret = data.monthly_total_returns(prices)
    assert mret.index.max() <= pd.Timestamp(data.AS_OF)
    q = st.sleeve_returns(mret, data.QUALITY)
    y = st.sleeve_returns(mret, data.YIELD)
    assert len(q) > 100 and len(y) > 100
