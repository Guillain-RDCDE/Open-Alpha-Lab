"""Strategy tests for Study 1016 — legs, reversibility, skew, mechanism, rules, verdict."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stairs import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(
    not data.have_real(),
    reason="bundled tapes not available (skfolio cache absent) — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Legs
# --------------------------------------------------------------------------- #
def test_zigzag_finds_the_hand_built_turning_points():
    # up 0.3 over 3 steps, down 0.3 over 1 step, up again
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4])
    ext, d0 = st.zigzag(p, 0.15)
    assert d0 == 1
    assert list(ext) == [0, 3, 4]


def test_zigzag_returns_nothing_on_a_quiet_path():
    ext, d0 = st.zigzag(np.zeros(50), 0.1)
    assert ext.size == 0 and d0 == 0


def test_first_passage_times_on_a_hand_built_path():
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4])
    L = st.legs(p, 0.15)
    assert list(L["dir"]) == [1, -1, 1]
    assert list(L["fp_days"]) == [2, 1, 2]       # up 2 steps to +0.15, down 1 step, up 2
    assert L["dur_days"].iloc[0] == 3 and L["dur_days"].iloc[1] == 1
    assert np.isnan(L["dur_days"].iloc[-1])       # final leg unconfirmed


def test_elevator_ratio_inverts_on_the_mirror_path(planted):
    """Negating the path swaps every drawup for a drawdown."""
    lp = np.concatenate([[0.0], np.cumsum(planted[0]["logret"].to_numpy())])
    a = st.leg_stats(lp, 0.05)
    b = st.leg_stats(-lp, 0.05)
    assert a["n_up"] == b["n_down"] and a["n_down"] == b["n_up"]
    assert a["elevator_ratio"] * b["elevator_ratio"] == pytest.approx(1.0, rel=0.05)


def test_thresholds_are_log_symmetric():
    # a 10% gain and a 10% "loss" are both a log move of log(1.1)
    p = np.array([0.0, np.log(1.1), 0.0, np.log(1.1), 0.0])
    L = st.legs(p, np.log1p(0.10) - 1e-12)
    assert set(L["fp_days"]) == {1}


def test_null_paths_quiet_on_the_matched_null(null_world):
    r = null_world[0]["logret"].to_numpy()
    t = st.leg_null_test(r, xs=(0.05,), kind="signflip", n_rep=80, seed=1)
    assert t.loc[0.05, "p"] > 0.05
    assert t.loc[0.05, "p_reverse"] > 0.05


def test_leverage_world_rebounds_faster_than_it_falls(planted):
    """The study's surprise, planted: leverage puts the volatility at the trough."""
    r = planted[0]["logret"].to_numpy()
    t = st.leg_null_test(r, xs=(0.05,), kind="signflip", n_rep=80, seed=1)
    assert t.loc[0.05, "elevator_ratio"] < 1.0
    assert t.loc[0.05, "p_reverse"] < 0.05


def test_signflip_null_preserves_absolute_returns():
    r = np.array([0.01, -0.02, 0.03, -0.005, 0.0])
    path = next(st._null_paths(r, "signflip", 1, seed=0))
    rr = np.diff(path)
    assert np.allclose(np.abs(rr - r.mean()), np.abs(r - r.mean()))


# --------------------------------------------------------------------------- #
# Time reversibility
# --------------------------------------------------------------------------- #
def test_tr_flips_sign_on_the_time_reversed_series(planted):
    x = st.standardise(planted[0]["logret"].to_numpy())
    assert st.tr_sum_fast(x[::-1]) == pytest.approx(-st.tr_sum_fast(x), rel=0.02)


def test_tr_fires_on_the_planted_world(planted):
    tr = st.tr_stats(planted[0]["logret"].to_numpy(), K=10, n_boot=150, seed=2)
    assert tr["tr_sum"] < 0
    assert tr["t_nw"] <= -2.0
    assert tr["t_boot"] <= -2.0


def test_tr_stays_quiet_on_the_null(null_world):
    tr = st.tr_stats(null_world[0]["logret"].to_numpy(), K=10, n_boot=150, seed=2)
    assert abs(tr["t_nw"]) < 2.0
    assert abs(tr["t_boot"]) < 2.0


def test_tr_is_zero_on_iid_gaussian_noise():
    r = np.random.default_rng(5).standard_normal(20000)
    tr = st.tr_stats(r, K=5, n_boot=50)
    assert abs(tr["t_nw"]) < 2.5
    assert tr["per_lag"].shape == (5, 2)


def test_nw_se_exceeds_naive_under_autocorrelation():
    rng = np.random.default_rng(0)
    e = np.zeros(4000)
    for t in range(1, 4000):
        e[t] = 0.6 * e[t - 1] + rng.normal()
    assert st.nw_tstat(e, lags=20)["se"] > st.nw_tstat(e, lags=0)["se"]


# --------------------------------------------------------------------------- #
# Skewness
# --------------------------------------------------------------------------- #
def test_skewness_signs_on_known_distributions():
    rng = np.random.default_rng(1)
    assert abs(st.skewness(rng.standard_normal(50000))) < 0.05
    assert st.skewness(rng.exponential(size=50000)) == pytest.approx(2.0, abs=0.2)
    assert st.skewness(-rng.exponential(size=50000)) < -1.5
    assert abs(st.kelly_skew(rng.standard_normal(50000))) < 0.02
    assert st.kelly_skew(rng.exponential(size=50000)) > 0.2


def test_block_bootstrap_interval_covers_the_estimate():
    x = np.random.default_rng(2).standard_normal(3000)
    ci = st.block_boot_ci(x, st.skewness, block=10, n_boot=200)
    assert ci["lo"] < ci["est"] < ci["hi"]
    assert ci["lo"] < 0 < ci["hi"]


def test_leverage_makes_monthly_returns_negatively_skewed(planted, null_world):
    def monthly(df):
        return st.aggregate(np.log(df["price"]), "ME").to_numpy()
    assert st.skewness(monthly(planted[0])) < st.skewness(monthly(null_world[0])) - 0.2


def test_aggregate_is_non_overlapping():
    s = pd.Series(np.arange(1, 61, dtype=float),
                  index=pd.bdate_range("2001-01-01", periods=60))
    m = st.aggregate(s, "ME")
    assert m.sum() == pytest.approx(s.resample("ME").last().iloc[-1]
                                    - s.resample("ME").last().iloc[0])


# --------------------------------------------------------------------------- #
# Mechanism
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def planted_fits(planted):
    return st.fit_models(planted[0]["logret"].to_numpy())


def test_gjr_fit_recovers_the_planted_leverage(planted_fits):
    g = planted_fits["gjr"]
    assert g["gamma"] == pytest.approx(data.PLANT["gamma"], abs=0.06)
    assert g["gamma_t"] > 3
    assert planted_fits["egarch"]["gamma"] < 0          # EGARCH's sign convention
    assert planted_fits["gjr"]["loglik"] > planted_fits["garch"]["loglik"]


def test_gjr_fit_finds_no_leverage_in_the_null(null_world):
    f = st.fit_models(null_world[0]["logret"].to_numpy())
    assert abs(f["gjr"]["gamma_t"]) < 2.5


def test_mechanism_check_credits_leverage_not_symmetry(planted, planted_fits):
    mc = st.mechanism_check(planted[0]["logret"].to_numpy(), planted_fits, n_paths=20,
                            seed=3)
    assert mc.loc["GJR-GARCH (leverage)", "share_tr"] > 0.4
    assert abs(mc.loc["GARCH (symmetric)", "share_tr"]) < 0.3
    assert abs(mc.loc["i.i.d. Student-t", "share_tr"]) < 0.2
    assert abs(mc.loc["i.i.d. shuffle of the tape", "share_tr"]) < 0.2


# --------------------------------------------------------------------------- #
# Rules and backtest
# --------------------------------------------------------------------------- #
def test_zigzag_rule_exits_and_reenters_on_the_thresholds():
    px = pd.Series([100, 110, 104, 104.4, 100, 105, 111, 112],
                   index=pd.bdate_range("2020-01-01", periods=8), dtype=float)
    sig = st.zigzag_rule(px, exit_x=0.05, entry_x=0.10)
    # peak 110, 104 is -5.45% -> out; trough 100, 111 is +11% -> back in
    assert list(sig) == [1, 1, 0, 0, 0, 0, 1, 1]


def test_backtest_all_in_equals_buy_and_hold():
    df, _ = data.synthetic_returns(1500, 1.0, seed=4)
    px = df["price"]
    rf = pd.Series(0.0001, index=px.index)
    bt = st.backtest(px, rf, pd.Series(1.0, index=px.index), cost=0.001)
    assert bt["sharpe_net"] == pytest.approx(bt["sharpe_bh"])
    assert bt["trades_per_year"] == 0.0 and bt["time_in"] == 1.0


def test_backtest_applies_one_lag_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.Series([100, 101, 50, 100, 101, 102], index=idx, dtype=float)
    rf = pd.Series(0.0, index=idx)
    sig = pd.Series([1, 1, 0, 0, 1, 1], index=idx, dtype=float)
    g = st.backtest(px, rf, sig, cost=0.0)
    n = st.backtest(px, rf, sig, cost=0.01)
    # the exit decided at the crash close (day 2) cannot dodge the crash itself
    assert g["ex_gross"].iloc[1] == pytest.approx(50 / 101 - 1)
    # ...it takes effect on the next return, and re-entry likewise waits one day
    assert g["ex_gross"].iloc[2] == pytest.approx(0.0)
    assert g["ex_gross"].iloc[3] == pytest.approx(0.0)
    assert g["ex_gross"].iloc[4] == pytest.approx(102 / 101 - 1)
    assert (g["ex_gross"] - n["ex_net"]).sum() == pytest.approx(0.02)


def test_sharpe_diff_test_is_zero_for_identical_series():
    x = pd.Series(np.random.default_rng(7).normal(0.0004, 0.01, 2000))
    t = st.sharpe_diff_test(x, x, block=20, n_boot=50)
    assert t["diff"] == 0.0 and t["lo"] == 0.0 and t["hi"] == 0.0


def test_race_reports_every_rule(planted):
    px = planted[0]["price"].iloc[:2500]
    rf = pd.Series(0.0001, index=px.index)
    out = st.race(px, rf, cost=0.0005, n_boot=40)
    assert set(out.index) == set(st.RULES)
    assert np.isnan(out.loc[st.PRIMARY_RULE, "p_vs_primary"])
    assert out["time_in"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"tr_sum": -2.8, "tr_t_nw": -3.4, "tr_t_boot": -4.0, "tr_t_nw_wins": -5.2,
         "nq_tr_t_nw": -3.6, "ff_tr_t_nw": -1.9,
         "elev05": 0.72, "elev10": 0.73, "elev20": 0.59, "gm_up10": 13.4, "gm_down10": 18.5,
         "elev10_p_signflip": 0.995, "elev10_p_reverse": 0.01, "elev10_p_perm": 1.0,
         "speed05": 1.49, "speed05_p": 0.01, "speed10": 1.76, "speed10_p": 0.07,
         "share_tr_gjr": 0.72, "share_elev_gjr": 1.06, "share_tr_egarch": 0.65,
         "share_elev_egarch": 1.26, "share_tr_garch": 0.05, "share_elev_garch": -0.44,
         "share_tr_iid": 0.0, "share_elev_iid": -0.05,
         "skew_d": -0.40, "skew_d_lo": -0.76, "skew_d_hi": -0.05, "skew_m": -0.74,
         "skew_m_lo": -1.07, "skew_m_hi": -0.36, "kelly_d": -0.043,
         "trade_d_vs_bh": {"ff": 0.14, "sp500": -0.30, "nasdaq": -0.22},
         "trade_ff_p_vs_bh": 0.045, "trade_ff_beats_sym": True, "trade_ff_post45_d": 0.08,
         "trade_ff_post45_p": 0.24, "trade_ff_sym10_d": 0.12,
         "cost_daily_bps": 5.0, "cost_monthly_bps": 10.0}
    h.update(over)
    return h


def test_verdict_signal_splits_by_leg():
    assert st.verdict(_headline())["signal"] == "Mixed"
    both = _headline(elev10=1.3, elev10_p_signflip=0.01)
    assert st.verdict(both)["signal"] == "Real"
    only_dur = _headline(tr_t_nw=-1.0, tr_t_boot=-1.2, elev10=1.3, elev10_p_signflip=0.01)
    assert st.verdict(only_dur)["signal"] == "Mixed"


def test_verdict_signal_weak_and_none():
    weak = _headline(tr_t_nw=-1.5, tr_t_boot=-1.7, elev10=1.1, elev10_p_signflip=0.3)
    assert st.verdict(weak)["signal"] == "Weak"
    none = _headline(tr_sum=0.2, tr_t_nw=0.3, tr_t_boot=0.4, elev10=0.9)
    assert st.verdict(none)["signal"] == "None"
    # robust t needs BOTH errors past 2
    half = _headline(tr_t_boot=-1.5)
    assert st.verdict(half)["signal"] == "None"     # rebound-faster tape, TR not certified


def test_verdict_tradability_in_both_directions():
    assert st.verdict(_headline())["trad"] == "Mirage"
    inv = _headline(trade_d_vs_bh={"ff": 0.2, "sp500": 0.1, "nasdaq": 0.05},
                    trade_ff_p_vs_bh=0.01)
    assert st.verdict(inv)["trad"] == "Investable"
    frag = _headline(trade_d_vs_bh={"ff": 0.2, "sp500": 0.1, "nasdaq": -0.05},
                     trade_ff_p_vs_bh=0.2)
    assert st.verdict(frag)["trad"] == "Fragile"
    no_sym = _headline(trade_d_vs_bh={"ff": 0.2, "sp500": 0.1, "nasdaq": 0.05},
                       trade_ff_p_vs_bh=0.01, trade_ff_beats_sym=False)
    assert st.verdict(no_sym)["trad"] == "Fragile"


def test_verdict_prose_matches_the_direction():
    v = st.verdict(_headline())
    assert "runs **backwards**" in v["signal_why"]
    assert "does not beat" in v["one_sentence"]
    v2 = st.verdict(_headline(elev10=1.3, elev10_p_signflip=0.01))
    assert "holds" in v2["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}


# --------------------------------------------------------------------------- #
# Real tape (gated) — the headline numbers, pinned loosely
# --------------------------------------------------------------------------- #
@needs_real
def test_real_sp500_is_time_irreversible_in_the_leverage_direction():
    r = st.log_returns(data.load_sp500()).to_numpy()
    tr = st.tr_stats(r, K=10, n_boot=100, seed=1016)
    assert tr["tr_sum"] < 0 and tr["t_nw"] <= -2.0 and tr["t_boot"] <= -2.0


@needs_real
def test_real_sp500_regains_ten_percent_faster_than_it_loses_it():
    lp = np.log(data.load_sp500().to_numpy())
    s = st.leg_stats(lp, 0.10)
    assert s["elevator_ratio"] < 1.0
    assert s["speed_ratio"] > 1.0
