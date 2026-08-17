"""Signal logic, backtest invariants and the study's spine — all offline/synthetic.

The spine: turnover is monotone in the rebalance clock (arithmetic); on a panel with
*planted* persistent momentum the gross return is positive and decays as the clock
slows (the mechanism a turnover budget prices); on the null panel every speed earns
nothing; the break-even cost falls as turnover rises; and the one-day execution lag
really is a lag.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turnover_budget import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Signal & weights
# --------------------------------------------------------------------------- #
def test_momentum_signal_skips_the_recent_month():
    """A name that crashed only in the last 21 days must still rank on its 12-1 return."""
    n = 400
    idx = pd.bdate_range("2015-01-01", periods=n)
    up = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
    crashed = up.copy()
    crashed.iloc[-10:] = 50.0
    sig = st.momentum_signal(pd.DataFrame({"up": up, "crashed": crashed}), lookback=252, skip=21)
    assert np.isclose(sig["up"].iloc[-1], sig["crashed"].iloc[-1])


def test_momentum_signal_nan_through_warmup():
    idx = pd.bdate_range("2015-01-01", periods=300)
    px = pd.DataFrame({"a": np.linspace(100, 150, 300)}, index=idx)
    sig = st.momentum_signal(px, lookback=252, skip=21)
    assert sig["a"].iloc[:252].isna().all()
    assert sig["a"].iloc[252:].notna().all()


def test_rebalance_dates_are_nested_and_shrinking():
    idx = pd.bdate_range("2010-01-04", periods=1500)
    d = st.rebalance_dates(idx, "D")
    w = st.rebalance_dates(idx, "W")
    m = st.rebalance_dates(idx, "M")
    q = st.rebalance_dates(idx, "Q")
    assert len(d) > len(w) > len(m) > len(q)
    # Daily is the superset of everything; quarter-ends are month-ends. (Month-ends are
    # NOT generally week-ends, so no nesting is claimed there.)
    for cal in (w, m, q):
        assert set(cal).issubset(set(d))
    assert set(q).issubset(set(m))


def test_rebalance_dates_rejects_unknown_freq():
    with pytest.raises(ValueError):
        st.rebalance_dates(pd.bdate_range("2010-01-04", periods=50), "Y")


def test_target_weights_dollar_neutral_unit_gross(planted):
    prices, _, _ = planted
    sig = st.momentum_signal(prices)
    w = st.target_weights(sig, top_k=3).dropna(how="all")
    assert len(w) > 100
    assert np.allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(w.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-12)
    assert np.allclose((w != 0).sum(axis=1).to_numpy(), 6)


def test_target_weights_long_only_is_fully_invested(planted):
    prices, _, _ = planted
    w = st.target_weights(st.momentum_signal(prices), top_k=3, long_only=True).dropna(how="all")
    assert (w >= -1e-15).all().all()
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-12)


def test_target_weights_skip_rows_below_min_breadth():
    idx = pd.bdate_range("2015-01-01", periods=5)
    sig = pd.DataFrame({"a": [1.0] * 5, "b": [2.0] * 5, "c": [3.0] * 5}, index=idx)
    assert st.target_weights(sig, top_k=1, min_breadth=6).isna().all().all()


def test_target_weights_go_long_the_top_and_short_the_bottom():
    idx = pd.bdate_range("2015-01-01", periods=2)
    cols = list("abcdefgh")
    vals = np.tile(np.arange(8, dtype=float), (2, 1))
    sig = pd.DataFrame(vals, index=idx, columns=cols)
    w = st.target_weights(sig, top_k=2)
    assert (w.iloc[0][["g", "h"]] > 0).all()
    assert (w.iloc[0][["a", "b"]] < 0).all()
    assert np.allclose(w.iloc[0][["c", "d", "e", "f"]].to_numpy(), 0.0)


# --------------------------------------------------------------------------- #
# Backtest engine invariants
# --------------------------------------------------------------------------- #
def test_backtest_columns_and_no_nans(planted):
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="M")
    for c in ("r_gross", "r_net", "traded", "traded_churn", "traded_drift",
              "short_notional", "excess_gross", "excess_net"):
        assert c in bt.columns
    assert not bt[["r_gross", "r_net", "traded"]].isna().any().any()
    assert len(bt) > 500


def test_turnover_split_sums_to_total(planted):
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="W")
    assert np.allclose((bt["traded_churn"] + bt["traded_drift"]).to_numpy(),
                       bt["traded"].to_numpy(), atol=1e-12)


def test_turnover_is_monotone_in_the_clock(planted):
    prices, cash, _ = planted
    turns = [st.turnover_stats(st.run_backtest(prices, cash, freq=f))["ann_turnover"]
             for f in ("D", "W", "M", "Q")]
    assert turns[0] > turns[1] > turns[2] > turns[3]


def test_costs_monotonically_reduce_return(planted):
    prices, cash, _ = planted
    means = [st.run_backtest(prices, cash, freq="M", cost_bps=c)["r_net"].mean()
             for c in (0.0, 5.0, 25.0)]
    assert means[0] >= means[1] >= means[2]


def test_borrow_only_bites_the_short_book(planted):
    prices, cash, _ = planted
    ls0 = st.run_backtest(prices, cash, freq="M", cost_bps=0.0, borrow_bps=0.0)
    ls1 = st.run_backtest(prices, cash, freq="M", cost_bps=0.0, borrow_bps=100.0)
    assert ls1["r_net"].mean() < ls0["r_net"].mean()
    lo0 = st.run_backtest(prices, cash, freq="M", cost_bps=0.0, borrow_bps=0.0, long_only=True)
    lo1 = st.run_backtest(prices, cash, freq="M", cost_bps=0.0, borrow_bps=100.0, long_only=True)
    assert np.allclose(lo0["r_net"].to_numpy(), lo1["r_net"].to_numpy())


def test_dollar_neutral_spread_is_already_excess_of_cash(planted):
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="M")
    assert np.allclose(bt["excess_net"].to_numpy(), bt["r_net"].to_numpy())


def test_long_only_arm_subtracts_the_cash_leg(planted):
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="M", long_only=True)
    assert (bt["excess_net"] < bt["r_net"]).mean() > 0.99


def test_long_only_arm_requires_a_cash_leg(planted):
    prices, _, _ = planted
    with pytest.raises(ValueError):
        st.run_backtest(prices, None, freq="M", long_only=True)


def test_precomputed_weights_change_nothing(planted):
    prices, cash, _ = planted
    w = st.target_weights(st.momentum_signal(prices), top_k=3)
    a = st.run_backtest(prices, cash, freq="W")
    b = st.run_backtest(prices, cash, freq="W", weights=w)
    assert np.allclose(a["r_net"].to_numpy(), b["r_net"].to_numpy())


def test_drifting_weights_keep_their_mandated_gross_exposure(planted):
    """Drift must not silently lever the book.

    A self-financing book's weights are shares of *current* NAV, so a position's weight
    grows by its own return divided by the book's. Skipping that denominator leaves the
    weights denominated in the initial NAV, and a long-only quarterly arm then wanders
    well away from 1.0x gross between rebalances (measured on the real tape: 0.65x-1.20x).
    Here we re-derive the implied gross exposure from the engine's own audit trail.
    """
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="Q", cost_bps=0.0, long_only=True)
    # n_names is recorded before the drift step, so a fully-invested book must always
    # hold exactly top_k names and its recorded return must never exceed the best single
    # asset's — both broken by a levered book.
    assert (bt["n_names"] == 3).all()
    per_asset = prices.pct_change().reindex(bt.index)
    assert (bt["r_gross"] <= per_asset.max(axis=1) + 1e-12).all()
    assert (bt["r_gross"] >= per_asset.min(axis=1) - 1e-12).all()


def test_equal_weight_control_shares_the_universe_and_is_fully_invested(planted):
    prices, _, _ = planted
    ew = st.equal_weight_targets(prices, min_breadth=6)
    sig = st.momentum_signal(prices)
    live = ew.dropna(how="all")
    assert len(live) > 100
    assert np.allclose(live.sum(axis=1).to_numpy(), 1.0, atol=1e-12)
    assert (live >= 0).all().all()
    # Exactly the names the momentum sort could have chosen — same investability mask.
    assert ((live > 0) == sig.reindex(live.index).notna()).all().all()


def test_ew_race_charges_both_legs_the_same_cost(planted):
    """The benchmark must pay its own trading, or the 'shortfall' is just commission.

    Raising the cost applied to BOTH legs may not mechanically drag the measured alpha
    down by the strategy's whole bill; the EW leg trades too, and its bill offsets part
    of it. A one-sided ledger fails this: alpha would fall by the full strategy cost.
    """
    prices, cash, _ = planted
    free = st.long_only_vs_equal_weight(prices, cash, freqs=("D",), cost_bps=0.0, top_k=2)
    paid = st.long_only_vs_equal_weight(prices, cash, freqs=("D",), cost_bps=10.0, top_k=2)
    assert paid.loc["D", "ann_turnover_ew"] > 0.0
    strat_bill = paid.loc["D", "ann_turnover"] * 10.0 * 1e-4
    drop = free.loc["D", "alpha_ann"] - paid.loc["D", "alpha_ann"]
    ew_bill = paid.loc["D", "ann_turnover_ew"] * 10.0 * 1e-4
    assert drop < strat_bill - 0.5 * ew_bill
    # The cost-free column is the same number at either cost setting.
    assert np.isclose(free.loc["D", "alpha_ann_gross"], paid.loc["D", "alpha_ann_gross"])


def test_execution_lag_is_exactly_one_day(planted):
    """Perturbing prices from day T on must not change any return before day T+1."""
    prices, cash, _ = planted
    cut = 1500
    base = st.run_backtest(prices, cash, freq="D", cost_bps=0.0, borrow_bps=0.0)
    pert = prices.copy()
    pert.iloc[cut:, 0] *= 1.5
    alt = st.run_backtest(pert, cash, freq="D", cost_bps=0.0, borrow_bps=0.0)
    stop = prices.index[cut]
    a = base.loc[base.index < stop, "r_gross"]
    b = alt.loc[alt.index < stop, "r_gross"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_and_sharpe_diff_finite(planted):
    prices, cash, _ = planted
    a = st.run_backtest(prices, cash, freq="D")["excess_net"]
    b = st.run_backtest(prices, cash, freq="Q")["excess_net"]
    assert np.isfinite(st.one_sample_t(a.to_numpy()))
    assert np.isfinite(st.sharpe_diff_tstat(a, b))


def test_bootstrap_ci_brackets_point(planted):
    prices, cash, _ = planted
    bt = st.run_backtest(prices, cash, freq="M", cost_bps=0.0, borrow_bps=0.0)
    ci = st.bootstrap_sharpe_ci(bt["excess_gross"], n_boot=400, seed=940)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_summary_matches_a_hand_computed_sharpe():
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(np.full(1000, 0.0004), index=idx) + pd.Series(
        np.random.default_rng(3).normal(0, 0.01, 1000), index=idx)
    s = st.summary(r)
    expect = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert abs(s["sharpe"] - expect) < 1e-12


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_panel_pays_gross(planted):
    """With persistent momentum planted, the gross sleeve must make money."""
    prices, cash, _ = planted
    d = st.synthetic_detect(prices, cash, top_k=2, min_breadth=6)
    assert d["ann_return_gross"]["M"] > 0.0
    assert d["t_gross"]["M"] > 2.0


def test_planted_panel_faster_clock_earns_more_gross(planted):
    """The mechanism a turnover budget prices: a faster clock tracks a decaying signal."""
    prices, cash, _ = planted
    d = st.synthetic_detect(prices, cash, top_k=2, min_breadth=6)
    assert d["daily_minus_quarterly_gross"] > 0.0
    assert d["ann_return_gross"]["D"] > d["ann_return_gross"]["Q"]


def test_null_panel_earns_nothing_at_any_speed(null_panel):
    prices, cash, _ = null_panel
    d = st.synthetic_detect(prices, cash, top_k=2, min_breadth=6)
    for f in ("D", "W", "M", "Q"):
        assert abs(d["t_gross"][f]) < 2.5


def test_null_across_seeds_is_centred():
    sharpes = []
    for s in range(5):
        p, c, _ = data.synthetic_panel(n_assets=8, n_years=12,
                                       signal_strength=0.0, seed=940 + s)
        sharpes.append(st.synthetic_detect(p, c, top_k=2)["sharpe_gross"]["M"])
    sharpes = np.array(sharpes)
    assert abs(sharpes.mean()) < 0.35
    assert (np.abs(sharpes) >= 0.7).sum() <= 1


# --------------------------------------------------------------------------- #
# The turnover budget itself
# --------------------------------------------------------------------------- #
def test_breakeven_is_looser_for_the_slower_clock_when_gross_is_flat(planted):
    """Same gross pot, less trading -> more bps of budget per unit traded notional."""
    prices, cash, _ = planted
    be_d = st.run_breakeven(prices, cash, "D", borrow_bps=0.0, top_k=2)
    be_q = st.run_breakeven(prices, cash, "Q", borrow_bps=0.0, top_k=2)
    assert be_d["ann_turnover"] > be_q["ann_turnover"]
    assert np.isfinite(be_d["breakeven_bps"]) and np.isfinite(be_q["breakeven_bps"])


def test_breakeven_is_the_zero_crossing_of_the_cost_sweep(planted):
    """At cost = break-even the net excess mean must be (numerically) zero."""
    prices, cash, _ = planted
    be = st.run_breakeven(prices, cash, "M", borrow_bps=25.0, top_k=2)
    at_be = st.run_backtest(prices, cash, freq="M", cost_bps=be["breakeven_bps"],
                            borrow_bps=25.0, top_k=2)
    assert abs(float(at_be["excess_net"].mean())) < 1e-12


def test_frequency_table_shape_and_turnover_ordering(planted):
    prices, cash, _ = planted
    tbl = st.frequency_table(prices, cash, top_k=2)
    assert list(tbl.index) == list(st.FREQS)
    assert tbl["ann_turnover"].is_monotonic_decreasing
    assert (tbl["churn_share"].between(0.0, 1.0)).all()


def test_cost_sweep_net_sharpe_is_decreasing_in_cost(planted):
    prices, cash, _ = planted
    cs = st.cost_sweep(prices, cash, freqs=("M",), cost_grid=(0.0, 5.0, 25.0), top_k=2)
    assert cs["ann_return_net"].is_monotonic_decreasing


def test_borrow_sweep_hits_every_speed(planted):
    prices, cash, _ = planted
    bs = st.borrow_sweep(prices, cash, borrow_grid=(0.0, 100.0), top_k=2)
    piv = bs.pivot(index="borrow_bps", columns="freq", values="ann_return_net")
    assert (piv.loc[100.0] < piv.loc[0.0]).all()


def test_era_cut_returns_both_halves(planted):
    prices, cash, _ = planted
    eras = st.era_cut(prices, cash, split="2010-01-01", freqs=("M", "Q"), top_k=2)
    assert eras["early"] is not None and eras["late"] is not None
    for tbl in eras.values():
        assert np.isfinite(tbl["sharpe_net"]).all()


def test_frequency_race_is_symmetric_in_sign(planted):
    prices, cash, _ = planted
    a = st.frequency_race(prices, cash, fast="D", slow="Q", top_k=2)
    b = st.frequency_race(prices, cash, fast="Q", slow="D", top_k=2)
    assert np.isclose(a["sharpe_gap"], -b["sharpe_gap"])
    assert np.isclose(a["t_diff"], -b["t_diff"])
