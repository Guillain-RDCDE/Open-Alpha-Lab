"""Strategy tests — offline, synthetic only. The detector must recover a planted trust
put and stay quiet when the put is a fiction; the mechanics (one lag, cost on NAV, the
excess-of-cash convention) must hold exactly."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trust_yield import data, strategy as st  # noqa: E402


def _spacs_from(meta):
    return tuple((tk, tk, str((meta.loc[tk, "deadline"] + pd.Timedelta(days=1)).date()))
                 for tk in meta.index)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_recovers_a_known_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 1.0, 4000)
    assert 2.0 < st.one_sample_t(x) < 5.0
    assert abs(st.one_sample_t(rng.normal(0.0, 1.0, 4000))) < 2.5


def test_newey_west_t_is_finite_and_shrinks_with_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(0.0, 1.0, 3000)
    ar = np.zeros(3000)
    for i in range(1, 3000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    ar = ar + 0.15
    t_iid = st.one_sample_t(ar)
    t_hac = st.newey_west_t(ar, lags=st.auto_lags(len(ar)))
    assert np.isfinite(t_hac)
    assert abs(t_hac) < abs(t_iid)


def test_implied_ytr_is_an_annualisation():
    assert abs(st.implied_ytr(10.0, 10.0, 365.25)) < 1e-12
    y = st.implied_ytr(9.50, 10.00, 365.25)
    assert abs(y - (10.0 / 9.5 - 1.0)) < 1e-9
    half = st.implied_ytr(9.50, 10.00, 365.25 / 2)
    assert half > y                                     # same discount, half the time


def test_summary_shapes():
    r = pd.Series(np.full(500, 0.0002))
    s = st.summary(r)
    assert s["n_days"] == 500 and s["sharpe"] > 0 and s["max_drawdown"] <= 0.0


# --------------------------------------------------------------------------- #
# Position mechanics — the honesty guarantees
# --------------------------------------------------------------------------- #
def test_positions_use_exactly_one_execution_lag(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    assert len(pos) > 50
    assert (pos["entry_date"] > pos["signal_date"]).all()
    for _, row in pos.head(40).iterrows():
        s = prices[row["ticker"]].dropna()
        i = s.index.get_loc(row["signal_date"])
        assert s.index[i + 1] == row["entry_date"]      # the very next close, never later
        assert abs(float(s.iloc[i + 1]) - row["px_entry"]) < 1e-9


def test_signal_uses_only_the_signal_day_quote(put_binds):
    """The buy decision must be made on the signal-day discount, not the entry-day one."""
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    assert (pos["discount_signal"] > 0).all()
    # Some entries will already have drifted back above trust by execution — proof that
    # the decision was not peeked forward.
    assert (pos["discount_entry"] <= 0).any() or (pos["discount_entry"] < pos["discount_signal"]).any()


def test_cost_is_one_way_on_nav_and_only_hurts():
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=10, n_days=650, seed=932)
    kw = dict(spacs=_spacs_from(meta), buffer_days=1, payoff=meta["payoff"].to_dict(),
              min_obs=40)
    free = st.build_positions(prices, cash, cost_bps=0.0, **kw)
    paid = st.build_positions(prices, cash, cost_bps=50.0, **kw)
    assert len(free) == len(paid)                        # cost never changes the signal
    assert paid["excess"].mean() < free["excess"].mean()
    # 50 bps one-way on NAV, charged once at entry.
    delta = (free["ret"] - paid["ret"]) / (1.0 + free["ret"])
    assert abs(float(delta.mean()) - 50e-4) < 5e-4


def test_the_daily_books_are_net_of_the_entry_cost():
    """A book run at 200 bps must be worse than the same book run gross.

    Regression guard: the books used to value positions off the raw fill while the
    position table was net of cost, so the reported exCAGR was silently a gross number
    and did not move at all across the cost sweep.
    """
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=8, n_days=650, seed=932)
    kw = dict(spacs=_spacs_from(meta), buffer_days=1, payoff=meta["payoff"].to_dict(),
              min_obs=40)
    free = st.build_positions(prices, cash, cost_bps=0.0, **kw)
    paid = st.build_positions(prices, cash, cost_bps=200.0, **kw)
    for mode in ("accrual", "mtm"):
        b_free = st.portfolio_daily(free, prices, cash, mode=mode)
        b_paid = st.portfolio_daily(paid, prices, cash, mode=mode)
        assert b_paid.mean() < b_free.mean(), mode
    assert (paid["px_entry_net"] > paid["px_entry"]).all()
    assert (free["px_entry_net"] == free["px_entry"]).all()


def test_the_headline_excess_is_an_identity_on_the_entry_discount(put_binds):
    """The measured excess IS the entry discount grossed up by cash — by construction.

    This is the study's central honesty claim, so it is asserted rather than asserted-in-
    prose: with the payoff imposed at the accrued trust and the benchmark the same cash
    leg the trust accrues at, no information about the redemption is being measured.
    """
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             min_obs=40)
    idr = st.identity_residual(pos)
    assert idr["n"] > 50
    assert idr["max_abs_residual"] < 1e-3
    assert idr["corr_with_discount"] > 0.99


def test_market_floor_never_pays_more_than_the_assumed_trust(put_binds):
    """The adversarial payoff must be a floor: weakly worse, never better."""
    prices, cash, meta, _ = put_binds
    kw = dict(spacs=_spacs_from(meta), buffer_days=1, min_obs=40)
    base = st.build_positions(prices, cash, **kw)
    floored = st.build_positions(prices, cash, market_floor=True, **kw)
    assert len(base) == len(floored)          # the payoff never changes the signal
    assert (floored["trust_exit"] <= base["trust_exit"] + 1e-12).all()
    assert floored["excess"].mean() <= base["excess"].mean() + 1e-12


def test_trust_anchor_check_is_one_row_per_name(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             min_obs=40)
    anc = st.trust_anchor_check(pos)
    assert len(anc) == pos["ticker"].nunique()
    assert set(anc.columns) == {"spac", "trust_exit", "px_exit_mkt", "gap"}
    assert anc["gap"].is_monotonic_increasing


def test_sell_leg_assumes_no_trust_payoff(put_binds):
    """``excess_sell`` must be built from the tape quote alone, cost paid both ways."""
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             min_obs=40, cost_bps=25.0)
    expect = pos["px_exit_mkt"] * (1 - 25e-4) / pos["px_entry_net"] - 1.0
    assert (pos["ret_sell"] - expect).abs().max() < 1e-12
    assert (pos["excess_sell"] - (pos["ret_sell"] - pos["cash_ret"])).abs().max() < 1e-12
    # and the redeem-or-sell bound is exactly the better of the two
    assert (pos["ret_best"] >= pos["ret"] - 1e-12).all()
    assert (pos["ret_best"] >= pos["ret_sell"] - 1e-12).all()


def test_excess_is_measured_against_cash_over_the_identical_window(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    row = pos.iloc[0]
    expect = float(cash.asof(row["exit_date"]) / cash.asof(row["entry_date"]) - 1.0)
    assert abs(row["cash_ret"] - expect) < 1e-12
    assert abs(row["excess"] - (row["ret"] - row["cash_ret"])) < 1e-12


def test_horizon_filter_is_respected(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40,
                             min_days=60, max_days=400)
    assert pos["days"].min() >= 60 and pos["days"].max() <= 400


def test_one_per_spac_takes_the_first_entry_only():
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=10, n_days=650, seed=932)
    kw = dict(spacs=_spacs_from(meta), buffer_days=1, payoff=meta["payoff"].to_dict(),
              min_obs=40)
    full = st.build_positions(prices, cash, **kw)
    one = st.build_positions(prices, cash, one_per_spac=True, **kw)
    assert len(one) == one["ticker"].nunique() <= full["ticker"].nunique()
    assert len(one) < len(full)


def test_no_position_is_taken_above_trust(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40,
                             max_discount=0.02)
    assert (pos["discount_signal"] > 0).all()
    assert (pos["discount_signal"] <= 0.02).all()


# --------------------------------------------------------------------------- #
# Daily books
# --------------------------------------------------------------------------- #
def test_daily_books_are_finite_and_excess_of_cash():
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=8, n_days=600, seed=932)
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    acc = st.portfolio_daily(pos, prices, cash, mode="accrual")
    mtm = st.portfolio_daily(pos, prices, cash, mode="mtm")
    assert len(acc) > 200 and len(mtm) > 200
    assert np.isfinite(acc.to_numpy()).all() and np.isfinite(mtm.to_numpy()).all()
    # The accrual book is a deterministic pull-to-payoff: far smoother than the quotes.
    assert acc.std() < mtm.std()
    assert acc.mean() > 0


def test_portfolio_daily_is_empty_on_no_positions(put_binds):
    prices, cash, _, _ = put_binds
    empty = pd.DataFrame(columns=["ticker"])
    assert len(st.portfolio_daily(empty, prices, cash)) == 0


# --------------------------------------------------------------------------- #
# The synthetic control: recover the plant, stay quiet on the null
# --------------------------------------------------------------------------- #
def test_detector_recovers_a_binding_trust_put(put_binds):
    prices, cash, meta, _ = put_binds
    out = st.synthetic_detect(prices, cash, meta)
    assert out["n_pos"] > 50
    assert out["mean_excess"] > 0.005
    assert out["ci_low"] > 0.0
    assert out["hit_rate"] > 0.8


def test_detector_still_trades_when_the_put_is_a_fiction(put_is_fiction):
    """The null must be a *live* world — the rule finds plenty to buy, it just earns nothing."""
    prices, cash, meta, _ = put_is_fiction
    out = st.synthetic_detect(prices, cash, meta)
    assert out["n_pos"] > 20
    assert np.isfinite(out["mean_excess"])


@pytest.mark.parametrize("seed", [932, 935, 938, 941])
def test_null_is_centred_across_seeds(seed):
    prices, cash, meta, _ = data.synthetic_panel(signal_strength=0.0, seed=seed, n_days=700)
    out = st.synthetic_detect(prices, cash, meta, n_boot=600)
    assert abs(out["mean_excess"]) < 0.06


@pytest.mark.parametrize("seed", [932, 935, 938, 941])
def test_plant_is_recovered_across_seeds(seed):
    prices, cash, meta, _ = data.synthetic_panel(signal_strength=1.0, seed=seed, n_days=700)
    out = st.synthetic_detect(prices, cash, meta, n_boot=600)
    assert out["ci_low"] > 0.0


def test_null_is_quiet_and_the_plant_is_loud_across_seeds():
    """Averaged over seeds: the plant fires every time, the null almost never.

    A single null panel can fire by luck — that is what a 95% interval means. The claim
    the study leans on is the *rate*, so the test is a rate, and it is asserted on the
    cluster bootstrap (the honest interval) rather than the position-level t.
    """
    seeds = range(932, 940)
    small = dict(n_days=700)
    loud = [st.synthetic_detect(*data.synthetic_panel(signal_strength=1.0, seed=s, **small)[:3],
                                n_boot=600)
            for s in seeds]
    quiet = [st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=s, **small)[:3],
                                 n_boot=600)
             for s in seeds]
    assert all(o["ci_low"] > 0.0 for o in loud)
    assert sum(o["ci_low"] > 0.0 for o in quiet) <= 2  # nominal 5%, 8 draws
    m_loud = np.mean([o["mean_excess"] for o in loud])
    m_quiet = np.mean([o["mean_excess"] for o in quiet])
    assert m_loud > 0.01
    assert abs(m_quiet) < 0.01
    assert m_loud > m_quiet


# --------------------------------------------------------------------------- #
# Bootstraps, era cut, sweeps
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_resamples_whole_names(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    b = st.cluster_bootstrap_mean(pos, n_boot=800, seed=1)
    assert b["n_clusters"] == pos["ticker"].nunique()
    assert b["ci_low"] < b["mean"] < b["ci_high"]


def test_cluster_bootstrap_is_wider_than_the_naive_interval(put_binds):
    """Monthly entries in one shell are one bet — the cluster CI must not be tighter."""
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    b = st.cluster_bootstrap_mean(pos, n_boot=2000, seed=2)
    x = pos["excess"].to_numpy(dtype=float)
    naive = 2 * 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert (b["ci_high"] - b["ci_low"]) > naive


def test_bootstrap_sharpe_ci_brackets_the_point():
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=8, n_days=600, seed=932)
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    e = st.portfolio_daily(pos, prices, cash, mode="mtm")
    ci = st.bootstrap_sharpe_ci(e, n_boot=400, seed=3)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_era_cut_splits_and_reports(put_binds):
    prices, cash, meta, _ = put_binds
    pos = st.build_positions(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                             payoff=meta["payoff"].to_dict(), min_obs=40)
    split = int(pos["year"].median())
    eras = st.era_cut(pos, split_year=split)
    got = [v for v in eras.values() if v is not None]
    assert len(got) == 2
    assert sum(v["n_pos"] for v in got) == len(pos)


def test_sweep_runs_a_grid():
    prices, cash, meta, _ = data.synthetic_panel(n_spacs=10, n_days=650, seed=932)
    rows = st.sweep(prices, cash, "cost_bps", (0.0, 100.0),
                    spacs=_spacs_from(meta), buffer_days=1,
                    payoff=meta["payoff"].to_dict(), min_obs=40)
    assert len(rows) == 2
    assert rows[0]["mean_excess"] > rows[1]["mean_excess"]


def test_yield_path_is_a_monthly_cross_section(put_binds):
    prices, cash, meta, _ = put_binds
    yp = st.yield_path(prices, cash, spacs=_spacs_from(meta), buffer_days=1,
                       max_discount=0.5)
    assert len(yp) > 5
    assert {"n_live", "n_below", "median_discount", "median_ytr"}.issubset(yp.columns)
    assert (yp["n_below"] <= yp["n_live"]).all()


def test_evaluate_on_no_positions_is_graceful():
    empty = pd.DataFrame(columns=["ticker", "excess"])
    out = st.evaluate(empty, pd.DataFrame(), pd.Series(dtype=float))
    assert out["n_pos"] == 0
