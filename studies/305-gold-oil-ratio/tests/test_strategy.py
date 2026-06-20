"""Signal construction, the timing-switch engine's invariants, the no-look-ahead lag, the
random-timing control, and the study's spine: the switch beats buy-and-hold and random timing
*only* when a real forward ratio→equity link is planted — and not on the null."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_oil_ratio import data, strategy as st  # noqa: E402


# ---- signal ---------------------------------------------------------------
def test_ratio_and_zscore_well_formed(null):
    daily, _ = null
    ratio = st.gold_oil_ratio(daily)
    assert (ratio > 0).all()
    z = st.ratio_zscore(daily, lookback=60)
    zf = z.dropna()
    assert len(zf) > 0
    # A standardised series is ~mean-zero, ~unit-sd over the sample.
    assert abs(zf.mean()) < 0.5
    assert 0.5 < zf.std() < 2.0


def test_zscore_warmup_is_nan(null):
    daily, _ = null
    z = st.ratio_zscore(daily, lookback=60)
    assert z.iloc[:59].isna().all()
    assert np.isfinite(z.iloc[60:]).any()


# ---- the switch & its one execution lag -----------------------------------
def test_positions_are_binary(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}


def test_position_is_lagged_one_day(null):
    """The held weight on day t is set from the z known at t-1 (one shift, applied once)."""
    daily, _ = null
    lookback, ez = 60, 1.0
    z = st.ratio_zscore(daily, lookback=lookback)
    raw = (z <= ez).astype(float).where(z.notna(), 1.0)
    pos = st.switch_positions(daily, lookback=lookback, enter_z=ez)
    # pos[t] should equal raw[t-1].
    assert np.allclose(pos.to_numpy()[1:], raw.to_numpy()[:-1])
    assert pos.iloc[0] == 1.0  # default-long seed


def test_high_ratio_means_cash(null):
    """A stretched-high ratio (z>enter_z) at t-1 must put the book in cash at t."""
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    z = st.ratio_zscore(daily, lookback=60)
    # Where z[t-1] > 1, pos[t] == 0.
    zlag = z.shift(1)
    cash_days = pos[(zlag > 1.0)]
    assert (cash_days == 0.0).all()


def test_random_positions_match_cash_fraction(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    rnd = st.random_positions(pos, seed=3)
    assert int((rnd == 0.0).sum()) == int((pos == 0.0).sum())
    # Reproducible.
    rnd2 = st.random_positions(pos, seed=3)
    assert np.array_equal(rnd.to_numpy(), rnd2.to_numpy())


# ---- returns engine invariants --------------------------------------------
def test_book_columns_and_cash_leg(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    book = st.book_returns(daily, pos, cost_bps=2.0)
    assert list(book.columns) == ["r_eq", "rf", "pos", "r_gross", "r_net", "r_bh"]
    # On a cash day the gross book return equals the risk-free rate.
    cash = book[book["pos"] == 0.0]
    if len(cash) > 0:
        assert np.allclose(cash["r_gross"], cash["rf"])


def test_net_is_gross_minus_switch_costs(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    book = st.book_returns(daily, pos, cost_bps=5.0)
    turnover = pos.reindex(book.index).fillna(1.0).diff().abs().fillna(0.0)
    expected = book["r_gross"] - turnover * 5e-4
    assert np.allclose(book["r_net"], expected)


def test_cost_monotonically_lowers_return(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    anns = [
        st.summarize_book(st.book_returns(daily, pos, cost_bps=c), "r_net")["ann_return"]
        for c in (0.0, 5.0, 20.0)
    ]
    assert anns[0] >= anns[1] >= anns[2]


def test_buy_hold_is_always_long():
    daily, _ = data.synthetic_daily(n_days=1000, pred_r=0.0, seed=1)
    bh = st.book_returns(daily, pd.Series(1.0, index=daily.index), cost_bps=5.0)
    assert (bh["pos"] == 1.0).all()
    # B&H gross return equals the equity return every day.
    assert np.allclose(bh["r_gross"], bh["r_eq"])


# ---- statistics -----------------------------------------------------------
def test_max_drawdown_sign_and_bounds(null):
    daily, _ = null
    pos = st.switch_positions(daily, lookback=60, enter_z=1.0)
    dd = st.max_drawdown(st.book_returns(daily, pos)["r_net"].to_numpy())
    assert -1.0 <= dd <= 0.0


def test_block_bootstrap_ci_orders():
    rng = np.random.default_rng(0)
    d = rng.normal(1e-4, 1e-3, 3000)
    lo, hi = st.block_bootstrap_ci(d, block=21, n_boot=500, seed=0)
    assert lo < hi


# ---- the study's spine ----------------------------------------------------
def test_switch_beats_benchmarks_only_when_signal_planted(null, planted):
    """On a folklore-consistent tape the switch clearly beats buy-and-hold AND random timing;
    on the null it does neither (it is a fair coin — |t| below the bar)."""
    r_null = st.race(null[0], lookback=60, enter_z=1.0, cost_bps=5.0)
    r_plant = st.race(planted[0], lookback=60, enter_z=1.0, cost_bps=5.0)

    # Planted: switch beats both, well above the bar.
    assert r_plant["t_vs_bh"] > 2.0
    assert r_plant["t_vs_random"] > 2.0

    # Null: not distinguishable from buy-and-hold.
    assert abs(r_null["t_vs_bh"]) < 2.0


def test_race_returns_full_structure(null):
    r = st.race(null[0], lookback=60, enter_z=1.0, cost_bps=5.0)
    for k in ("switch", "buy_hold", "random", "t_vs_bh", "t_vs_random",
              "ci_vs_bh", "ci_vs_random", "n_switches"):
        assert k in r
    for book in ("switch", "buy_hold", "random"):
        for stat in ("ann_return", "sharpe_excess", "max_dd", "pct_long", "t_excess"):
            assert stat in r[book]
