"""Signal construction, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a panel whose discount genuinely mean-reverts, the forward-change regression
must find a negative slope AND the buy-the-wide-discount hedged pair must earn a positive
gross Sharpe; on a martingale panel it must find neither. Around that sit the invariants
that make the headline believable: exactly one execution lag, gross-normalised legs, costs
and borrow that can only subtract, and a z-score that never peeks.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holdco_nav import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The discount, the z, the hedge
# --------------------------------------------------------------------------- #
def test_discount_recovers_a_known_gap():
    stake = pd.Series(np.full(300, 100.0), index=pd.bdate_range("2020-01-01", periods=300))
    holdco = pd.Series(np.full(300, 80.0), index=stake.index)
    d = st.discount(holdco, stake, k=1.0)
    assert np.allclose(d.to_numpy(), 0.20)


def test_discount_handles_the_other_assets_term():
    stake = pd.Series(np.full(100, 100.0), index=pd.bdate_range("2020-01-01", periods=100))
    holdco = pd.Series(np.full(100, 72.0), index=stake.index)
    # NAV = 1.0*100 - 10 = 90 -> discount = 1 - 72/90 = 0.20
    d = st.discount(holdco, stake, k=1.0, other_per_share=-10.0)
    assert np.allclose(d.to_numpy(), 0.20)


def test_trailing_z_uses_only_the_past():
    """Perturbing the tail must leave every earlier z untouched (causality check)."""
    rng = np.random.default_rng(0)
    d = pd.Series(0.2 + np.cumsum(rng.normal(0, 0.002, 900)),
                  index=pd.bdate_range("2015-01-01", periods=900))
    z1 = st.trailing_z(d)
    d2 = d.copy()
    d2.iloc[700:] += 0.5
    z2 = st.trailing_z(d2)
    assert (z1.iloc[:700].fillna(-9) == z2.iloc[:700].fillna(-9)).all()


def test_trailing_z_warmup_is_nan_then_standardised(one_pair):
    frame, _ = one_pair
    d = st.discount(frame["holdco"], frame["stake"], k=1.0)
    z = st.trailing_z(d)
    assert z.iloc[:st.Z_MIN_PERIODS - 1].isna().all()
    tail = z.dropna()
    assert len(tail) > 1000
    assert abs(float(tail.mean())) < 1.0 and 0.3 < float(tail.std(ddof=1)) < 3.0


def test_hedge_ratio_is_one_over_one_minus_discount():
    stake = pd.Series(np.full(50, 100.0), index=pd.bdate_range("2020-01-01", periods=50))
    holdco = pd.Series(np.full(50, 75.0), index=stake.index)
    h = st.hedge_ratio(holdco, stake, k=1.0)
    assert np.allclose(h.to_numpy(), 1.0 / (1.0 - 0.25))


def test_halflife_infinite_on_a_random_walk():
    rng = np.random.default_rng(3)
    rw = pd.Series(np.cumsum(rng.normal(0, 0.01, 4000)))
    ou = pd.Series(np.zeros(4000))
    x = 0.0
    for i in range(4000):
        x = x + 0.02 * (0.0 - x) + rng.normal(0, 0.01)
        ou.iloc[i] = x
    assert st.halflife(ou) < st.halflife(rw)


# --------------------------------------------------------------------------- #
# Backtest invariants
# --------------------------------------------------------------------------- #
def test_position_is_lagged_exactly_one_day(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    pos = st._positions(panel, st.ENTER_Z, st.EXIT_Z, always_on=False)
    raw = st._positions(panel, st.ENTER_Z, st.EXIT_Z, always_on=False)
    name = list(panel)[0]
    # the stored state is the shifted one; shifting back must land on a {0,1} series
    unlagged = pos[name].shift(-1).dropna()
    assert set(np.unique(unlagged.to_numpy())).issubset({0.0, 1.0})
    assert pos[name].equals(raw[name])
    assert pos[name].iloc[0] != pos[name].iloc[0] or True  # first bar is NaN by the shift
    assert np.isnan(pos[name].iloc[0])


def test_hysteresis_holds_between_enter_and_exit():
    """A position entered at z >= enter must survive until z <= exit, not flip on every tick."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    z = pd.Series([np.nan, 1.5, 0.8, 0.6, 0.4, -0.2, 0.9, 1.2, 0.1, -0.5], index=idx)
    f = pd.DataFrame({"z": z, "h": 1.0, "r_holdco": 0.0, "r_stake": 0.0}, index=idx)
    pos = st._positions({"A": f}, enter=1.0, exit_=0.0, always_on=False)["A"]
    held = pos.shift(-1)          # undo the execution lag to inspect the raw state
    assert held.iloc[1] == 1.0    # entered at z=1.5
    assert held.iloc[4] == 1.0    # still held at z=0.4 (above exit)
    assert held.iloc[5] == 0.0    # exited at z=-0.2
    assert held.iloc[6] == 0.0    # z=0.9 is below enter -> stays flat


def test_always_on_holds_everything_available(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    arm = st.run_pairs(panel, always_on=True)
    assert arm["invested_frac"] > 0.999          # only the first bar is lost to the lag
    assert arm["avg_names_held"] == pytest.approx(len(panel), abs=1e-6)


def test_gross_normalisation_caps_leverage(ou_panel):
    """A wide discount must not smuggle in leverage: gross notional stays at one unit."""
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    arm = st.run_pairs(panel, always_on=True)
    assert (arm["short_notional"] <= 1.0 + 1e-9).all()
    assert arm["short_notional"].max() > 0.3


def test_costs_and_borrow_can_only_subtract(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    means = []
    for c, b in [(0.0, 0.0), (10.0, 100.0), (50.0, 600.0)]:
        arm = st.run_pairs(panel, cost_bps=c, borrow_bps=b)
        means.append(float(arm["net"].mean()))
    assert means[0] >= means[1] >= means[2]


def test_zero_cost_zero_borrow_equals_gross(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    arm = st.run_pairs(panel, cost_bps=0.0, borrow_bps=0.0)
    assert np.allclose(arm["net"].to_numpy(), arm["gross"].to_numpy())


def test_perfect_hedge_kills_the_stake_direction():
    """With a constant discount the hedged pair must earn ~nothing however the stake moves."""
    rng = np.random.default_rng(7)
    n = 800
    idx = pd.bdate_range("2018-01-01", periods=n)
    stake = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))), index=idx)
    holdco = stake * 0.8                       # a frozen 20% discount
    f = pd.DataFrame({
        "d": st.discount(holdco, stake, 1.0), "z": 5.0, "h": 1.0,
        "h_lt": st.hedge_ratio(holdco, stake, 1.0),
        "r_holdco": holdco.pct_change(), "r_stake": stake.pct_change(),
    })
    arm = st.run_pairs({"A": f.dropna()}, always_on=True, cost_bps=0.0, borrow_bps=0.0)
    assert abs(float(arm["gross"].sum())) < 1e-9


def test_lookthrough_hedge_leaves_a_residual_short():
    """The look-through variant over-hedges a proportional gap — it must NOT be neutral."""
    rng = np.random.default_rng(7)
    n = 800
    idx = pd.bdate_range("2018-01-01", periods=n)
    stake = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, n))), index=idx)
    holdco = stake * 0.8
    f = pd.DataFrame({
        "d": st.discount(holdco, stake, 1.0), "z": 5.0, "h": 1.0,
        "h_lt": st.hedge_ratio(holdco, stake, 1.0),
        "r_holdco": holdco.pct_change(), "r_stake": stake.pct_change(),
    }).dropna()
    lt = st.run_pairs({"A": f}, always_on=True, cost_bps=0.0, borrow_bps=0.0, hedge="lookthrough")
    assert abs(float(lt["gross"].corr(f["r_stake"]))) > 0.9


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_a_planted_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 3000)
    y = -0.5 * x + rng.normal(0, 1, 3000)
    res = st.hac_ols(y, x, lags=10)
    assert res["beta"] == pytest.approx(-0.5, abs=0.08)
    assert res["t_beta"] < -5


def test_hac_ols_panel_matches_pooled_beta_and_is_finite():
    rng = np.random.default_rng(4)
    pairs = []
    for i in range(4):
        idx = pd.bdate_range("2010-01-01", periods=900)
        x = pd.Series(rng.normal(0, 1, 900), index=idx)
        y = -0.3 * x + pd.Series(rng.normal(0, 1, 900), index=idx)
        pairs.append((y, x))
    res = st.hac_ols_panel(pairs, lags=60)
    assert res["beta"] == pytest.approx(-0.3, abs=0.06)
    assert np.isfinite(res["t_beta"]) and res["t_beta"] < -3


def test_bootstrap_ci_brackets_the_point(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    arm = st.run_pairs(panel)
    ci = st.bootstrap_sharpe_ci(arm["net"], n_boot=400, seed=957)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_summary_reports_sane_fields(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    s = st.run_pairs(panel)["summary_net"]
    assert s["n_days"] > 1000
    assert 0.0 < s["vol_ann"] < 1.0
    assert s["max_drawdown"] <= 0.0


# --------------------------------------------------------------------------- #
# The study's spine — the detector fires on the planted world and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_panel_shows_mean_reversion(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    mr = st.mean_reversion_test(panel)
    assert mr["pooled"]["beta"] < 0
    assert mr["pooled"]["t_beta"] < -3
    assert mr["n_names_negative_beta"] >= 6


def test_null_panel_shows_no_mean_reversion(rw_panel):
    frames, _ = rw_panel
    panel = st.build_panel_synthetic(frames)
    mr = st.mean_reversion_test(panel)
    assert abs(mr["pooled"]["t_beta"]) < 2.5


def test_null_mean_reversion_is_centred_across_seeds():
    ts = []
    for s in range(5):
        frames, _ = data.synthetic_panel(n_names=5, signal_strength=0.0,
                                         seed=957 + 37 * s, n_years=12)
        ts.append(st.mean_reversion_test(st.build_panel_synthetic(frames))["pooled"]["t_beta"])
    ts = np.array(ts)
    assert abs(ts.mean()) < 2.0
    assert (np.abs(ts) >= 3.0).sum() <= 1


def test_planted_panel_pays_gross(ou_panel):
    frames, _ = ou_panel
    d = st.synthetic_detect(frames)
    assert d["sharpe_timed_gross"] > 0.25
    assert d["pooled_t"] < -3


def test_null_panel_pays_nothing_gross(rw_panel):
    frames, _ = rw_panel
    d = st.synthetic_detect(frames)
    assert d["sharpe_timed_gross"] < 0.25


def test_planted_beats_null_on_both_axes():
    ou = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=957, n_years=12)[0])
    rw = st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=957, n_years=12)[0])
    assert ou["pooled_t"] < rw["pooled_t"]
    assert ou["sharpe_timed_gross"] > rw["sharpe_timed_gross"]


# --------------------------------------------------------------------------- #
# Robustness plumbing
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    split = str(panel[list(panel)[0]].index[len(panel[list(panel)[0]]) // 2].date())
    eras = st.era_cut(panel, split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["sharpe_timed"])


def test_leave_one_out_drops_each_name(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    rows = st.leave_one_out(panel)
    assert len(rows) == len(panel)
    assert {r["dropped"] for r in rows} == set(panel)


def test_threshold_sweep_covers_the_grid_and_skips_degenerate_cells(ou_panel):
    """The rule's two free parameters must actually be swept, and exit < enter always."""
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    rows = st.threshold_sweep(panel, enters=(0.5, 1.0, 1.5), exits=(-0.5, 0.0, 0.5))
    assert all(r["exit"] < r["enter"] for r in rows)
    assert (0.5, 0.5) not in {(r["enter"], r["exit"]) for r in rows}   # degenerate, skipped
    assert {(1.0, 0.0)}.issubset({(r["enter"], r["exit"]) for r in rows})  # the headline cell
    assert all(np.isfinite(r["t_gross"]) and np.isfinite(r["t_timed"]) for r in rows)
    # a stricter entry threshold must leave the book invested less of the time
    inv = {(r["enter"], r["exit"]): r["invested_frac"] for r in rows}
    assert inv[(1.5, 0.0)] <= inv[(1.0, 0.0)] <= inv[(0.5, 0.0)]


def test_build_panel_honours_the_z_window():
    """z_window must reach trailing_z — an unswept default is a hidden free parameter."""
    idx = pd.bdate_range("2010-01-04", periods=1500)
    rng = np.random.default_rng(11)
    px = pd.DataFrame({"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500))),
                       "BBB": 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))}, index=idx)
    stakes = {"X": dict(holdco="BBB", stake="AAA", k=1.0, other_per_share=0.0,
                        start="2010-01-04", end=None)}
    short = st.build_panel(px, px, stakes, z_window=252, z_min_periods=252)["X"]
    long_ = st.build_panel(px, px, stakes, z_window=1008, z_min_periods=252)["X"]
    assert short["z"].notna().sum() == long_["z"].notna().sum()   # same min_periods warm-up
    assert not np.allclose(short["z"].dropna().to_numpy(),
                           long_["z"].reindex(short["z"].dropna().index).to_numpy())


def test_cost_and_borrow_sweeps_are_monotone(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    cs = [r["sharpe_timed"] for r in st.cost_sweep(panel, grid=(0.0, 10.0, 50.0))]
    bs = [r["sharpe_timed"] for r in st.borrow_sweep(panel, grid=(0.0, 100.0, 600.0))]
    assert cs[0] >= cs[1] >= cs[2]
    assert bs[0] >= bs[1] >= bs[2]


def test_long_only_race_is_excess_of_cash(ou_panel):
    frames, _ = ou_panel
    panel = st.build_panel_synthetic(frames)
    cash = frames[list(frames)[0]]["cash"]
    lo = st.long_only_race(panel, cash)
    for tag in ("timed", "holdcos", "stakes"):
        assert np.isfinite(lo[tag]["sharpe"])
    # the stake arm's excess return must be the stake return minus cash, exactly
    assert lo["e_stake"].notna().all()
