"""Strategy tests for Study 1023 — the tether, the break, and the trader who believed in both.

Synthetic-first: every detector must fire on the planted world and stay quiet on the null.
Real-tape checks are gated on the arch tape being importable.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cushing import data, strategy as st  # noqa: E402
from quantlab.analytics import mean_tstat_hac  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes unavailable — synthetic tests cover the logic")


def _lr(px):
    return st.spread_frame(px)["log_ratio"]


# --------------------------------------------------------------------------- #
# The spread
# --------------------------------------------------------------------------- #
def test_spread_frame_identities(planted):
    px, _ = planted
    sf = st.spread_frame(px)
    assert np.allclose(sf["spread"], px["brent"] - px["wti"])
    assert np.allclose(np.exp(sf["log_ratio"]), px["brent"] / px["wti"])


# --------------------------------------------------------------------------- #
# Stationarity
# --------------------------------------------------------------------------- #
def test_adf_rejects_on_the_planted_tether(planted):
    assert st.adf(_lr(planted[0]))["p"] < 0.01


def test_adf_stays_quiet_on_the_random_walk(null_world):
    assert st.adf(_lr(null_world[0]))["p"] > 0.10


def test_kpss_accepts_planted_and_rejects_null(planted, null_world):
    assert not st.kpss_test(_lr(planted[0]))["reject5"]
    assert st.kpss_test(_lr(null_world[0]))["reject5"]


def test_joint_reading_four_cells():
    yes, no = {"reject5": True}, {"reject5": False}
    assert st.joint_reading(yes, no) == "stationary"
    assert st.joint_reading(no, yes) == "unit root"
    assert st.joint_reading(yes, yes).startswith("conflict")
    assert st.joint_reading(no, no).startswith("inconclusive")
    assert st.joint_reading({}, no) == "n/a"


def test_stationarity_table_reads_both_series(planted):
    sf = st.spread_frame(planted[0])
    t = st.stationarity_table(sf, [("all", None, None), ("first half", None, "2003-12-31")])
    assert set(t["series"]) == {"spread", "log_ratio"}
    assert len(t) == 4
    assert (t.loc[(t.window == "all") & (t.series == "log_ratio"), "reading"]
            == "stationary").all()


def test_engle_granger_finds_the_planted_cointegration(planted, null_world):
    a = st.engle_granger(planted[0])
    assert a["reject5"] and a["beta"] == pytest.approx(1.0, abs=0.05)
    assert not st.engle_granger(null_world[0])["reject5"]


def test_short_inputs_decline():
    assert st.adf(np.arange(10.0)) == {}
    assert st.kpss_test(np.arange(10.0)) == {}
    assert st.ar1_fit(np.arange(10.0)) == {}


# --------------------------------------------------------------------------- #
# Breaks
# --------------------------------------------------------------------------- #
def test_sup_f_locates_the_planted_break(broken):
    px, tr = broken
    s = st.sup_f_mean(_lr(px))
    gap = abs(px.index.get_loc(s["break_date"]) - tr["break_index"])
    assert gap <= 6
    assert s["mean_after"] - s["mean_before"] == pytest.approx(0.15, abs=0.04)


def test_bootstrap_break_test_fires_on_break_and_not_without(broken, planted):
    assert st.sup_f_bootstrap(_lr(broken[0]), n_boot=99)["p"] < 0.05
    assert st.sup_f_bootstrap(_lr(planted[0]), n_boot=99)["p"] > 0.05


def test_andrews_critical_values_are_too_lenient_on_a_persistent_series(planted):
    """The reason the bootstrap exists: a break-free OU spread already clears the
    asymptotic 5% value, which assumes weak dependence."""
    b = st.sup_f_bootstrap(_lr(planted[0]), n_boot=99)
    assert b["null_q95"] > st.ANDREWS_CV_15["5%"]


def test_cusum_fires_on_break_not_on_tether(broken, planted):
    assert st.cusum_mean(_lr(broken[0]))["reject5"]
    assert not st.cusum_mean(_lr(planted[0]))["reject5"]


def test_bai_perron_one_break_solution_is_the_planted_one(broken):
    px, tr = broken
    bp = st.bai_perron(_lr(px), max_breaks=3, min_seg=24)
    one = bp["solutions"][1][0]
    assert abs(px.index.get_loc(one) - tr["break_index"]) <= 6
    assert bp["m_lwz"] >= 1
    assert bp["table"]["ssr"].is_monotonic_decreasing


def test_bai_perron_count_overstates_but_the_spurious_shifts_are_small(planted, broken):
    """Documented caveat: with errors this persistent, LWZ finds 'breaks' in a break-free
    tether. They are tiny next to a real one — which is why the study takes its p-value
    from the bootstrapped sup-F and uses Bai–Perron only to locate."""
    def biggest_jump(px):
        segs = st.bai_perron(_lr(px), max_breaks=3)["segments"]
        m = [s["mean"] for s in segs]
        return max(abs(np.diff(m))) if len(m) > 1 else 0.0
    assert biggest_jump(planted[0]) < 0.06
    assert biggest_jump(broken[0]) > 0.10


def test_bai_perron_respects_minimum_segment(broken):
    bp = st.bai_perron(_lr(broken[0]), max_breaks=4, min_seg=30)
    for s in bp["segments"]:
        assert s["n"] >= 30


# --------------------------------------------------------------------------- #
# Half-life
# --------------------------------------------------------------------------- #
def test_ar1_recovers_the_planted_half_life():
    px, tr = data.synthetic_spread(n_months=3000, seed=11)
    f = st.ar1_fit(_lr(px))
    assert f["halflife_lo"] < tr["halflife_months"] < f["halflife_hi"]
    assert f["phi"] == pytest.approx(tr["phi"], abs=0.03)


def test_ar1_half_life_on_random_walk_is_long_or_infinite(null_world):
    f = st.ar1_fit(_lr(null_world[0]))
    assert f["halflife"] > 12 and np.isinf(f["halflife_hi"])


def test_halflife_by_regime_splits_at_the_given_dates(broken):
    px, tr = broken
    t = st.halflife_by_regime(_lr(px), [tr["break_date"]])
    assert len(t) == 2
    assert t["n"].sum() == len(px)
    assert t["mean"].iloc[1] > t["mean"].iloc[0]


# --------------------------------------------------------------------------- #
# The trader
# --------------------------------------------------------------------------- #
def test_realtime_z_has_no_lookahead(planted):
    x = _lr(planted[0])
    z1 = st.zscore_realtime(x)
    x2 = x.copy()
    x2.iloc[200:] += 5.0          # vandalise the future
    z2 = st.zscore_realtime(x2)
    assert np.allclose(z1.iloc[:200].dropna(), z2.iloc[:200].dropna())
    zr1 = st.zscore_realtime(x, window=60)
    zr2 = st.zscore_realtime(x2, window=60)
    assert np.allclose(zr1.iloc[:200].dropna(), zr2.iloc[:200].dropna())
    assert z1.iloc[:35].isna().all()


def test_frozen_z_is_undefined_inside_its_calibration_window(planted):
    z = st.zscore_frozen(_lr(planted[0]), calib_end="2009-12-31")
    assert z[:"2009-12-31"].isna().all()
    assert z["2010-01-31":].notna().all()


def test_positions_state_machine():
    z = pd.Series([0.0, 2.5, 1.5, 0.4, -2.2, -1.0, 3.0, np.nan, 0.0])
    p = st.positions(z, entry=2.0, exit=0.5).tolist()
    assert p == [0, -1, -1, 0, 1, 1, -1, 0, 0]


def test_book_applies_exactly_one_lag_and_charges_costs():
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    px = pd.DataFrame({"brent": [100, 110, 110, 121, 121.0],
                       "wti": [100, 100, 100, 100, 100.0]}, index=idx)
    pos = pd.Series([0, 1, 1, 0, 0.0], index=idx)
    b = st.book(px, pos, cost_bps=10, roll_bps=0)
    # the +10% Brent move in month 1 is NOT earned: the position is only set at month 1
    assert b.loc[idx[1], "gross"] == 0.0
    assert b.loc[idx[2], "gross"] == 0.0           # Brent flat in month 2
    assert b.loc[idx[3], "gross"] == pytest.approx(0.10)
    assert b.loc[idx[2], "trade_cost"] == pytest.approx(2 * 10 / 1e4)
    assert b.loc[idx[4], "trade_cost"] == pytest.approx(2 * 10 / 1e4)
    assert (b["net"] <= b["gross"] + 1e-15).all()
    assert b.loc[idx[3], "usd"] == pytest.approx(11.0)


def test_roll_toll_charged_only_while_holding():
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    px = pd.DataFrame({"brent": [100.0] * 4, "wti": [100.0] * 4}, index=idx)
    pos = pd.Series([1, 1, 0, 0.0], index=idx)
    b = st.book(px, pos, cost_bps=0, roll_bps=3)
    assert b["roll_cost"].tolist() == pytest.approx([6e-4, 6e-4, 0.0])


def test_the_fader_makes_money_on_a_tether_and_not_on_a_random_walk():
    """Positive control and null, averaged over seeds so one lucky path cannot carry it."""
    tp = st.synthetic_trade_power(strengths=(0.0, 1.0), n_reps=12)
    assert tp.loc[1.0, "median_t"] > 2.0
    assert tp.loc[0.0, "share_t_ge_2"] <= 0.1


def test_the_frozen_trader_bleeds_through_a_planted_break(broken):
    px, tr = broken
    x = _lr(px)
    calib_end = str(px.index[tr["break_index"] - 24].date())
    bk = st.book(px, st.positions(st.zscore_frozen(x, calib_end=calib_end)))
    ep = st.episodes(bk)
    assert (ep["side"] == "short spread").any()
    assert ep["mae"].min() < -0.05


def test_episodes_and_mae_are_consistent(planted):
    px, _ = planted
    bk = st.book(px, st.positions(st.zscore_realtime(_lr(px))))
    ep = st.episodes(bk)
    assert len(ep) > 3
    assert (ep["mae"] <= 0).all()
    assert (ep["mae"] <= ep["net_return"] + 1e-12).all()
    assert (ep["months"] >= 1).all()


def test_underwater_on_a_hand_built_path():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    r = pd.Series([0.10, -0.05, -0.05, 0.20, -0.01, 0.0], index=idx)
    u = st.underwater(r)
    assert u["months"] == 2 and u["peak"] == idx[0] and u["recovered"] == idx[3]
    u2 = st.underwater(pd.Series([0.1, -0.2, 0.01, 0.01], index=idx[:4]))
    assert u2["recovered"] is None and u2["months"] == 3


def test_perf_reports_hac_and_bootstrap(planted):
    px, _ = planted
    bk = st.book(px, st.positions(st.zscore_realtime(_lr(px))))
    p = st.perf(bk["net"], n_boot=200)
    assert p["sr_lo"] <= p["sharpe"] <= p["sr_hi"]
    assert np.isfinite(p["t_hac"]) and p["maxdd"] <= 0
    assert p["t_hac"] == pytest.approx(mean_tstat_hac(bk["net"])["tstat"])


def test_costs_only_ever_hurt(planted):
    px, _ = planted
    pos = st.positions(st.zscore_realtime(_lr(px)))
    cs = st.cost_sweep(px, pos, grid=(0, 10, 50))
    assert cs["ann"].is_monotonic_decreasing
    assert st.break_even_cost(px, pos) > 0
    assert np.isinf(st.break_even_cost(px, pos * 0))


# --------------------------------------------------------------------------- #
# Power — what a break does to ADF
# --------------------------------------------------------------------------- #
def test_a_level_break_masquerades_as_a_unit_root():
    clean = st.power_curve(strengths=(1.0,), n_reps=20)
    brk = st.power_curve(strengths=(1.0,), n_reps=20, break_size=0.15)
    assert clean.loc[1.0, "adf_reject"] >= 0.9
    assert brk.loc[1.0, "adf_reject"] < clean.loc[1.0, "adf_reject"] - 0.3
    assert brk.loc[1.0, "kpss_reject"] >= 0.9


def test_adf_power_falls_as_reversion_slows():
    pw = st.power_curve(strengths=(0.0, 0.1, 1.0), n_reps=20)
    assert pw.loc[0.0, "adf_reject"] <= 0.15
    assert pw.loc[0.1, "adf_reject"] < pw.loc[1.0, "adf_reject"]


# --------------------------------------------------------------------------- #
# The verdict rule, both directions
# --------------------------------------------------------------------------- #
def _h(**over):
    h = {"n_months": 393, "t_gross": 1.1, "t_net": 0.8, "adf_p_full": 0.45,
         "kpss_p_full": 0.01, "break_p": 0.002, "break_date": "2010-09", "sup_f": 766.0,
         "adf_p_pre": 0.066, "adf_p_pre_usd": 1e-9, "hl_pre": 2.7, "hl_pre_lo": 1.9,
         "hl_pre_hi": 4.2, "mean_pre_usd": -1.4, "mean_post_usd": 7.7,
         "sr_gross_lo": -0.09, "sr_gross_hi": 0.38, "ann_net": 0.009, "sr_net": 0.12,
         "frozen_entry": "2010-05", "frozen_mae": -0.24, "frozen_mae_usd": -25.1,
         "frozen_underwater_months": 114, "frozen_recovered": False}
    h.update(over)
    return h


def test_verdict_real_needs_everything_at_once():
    real = _h(t_gross=2.5, adf_p_full=0.01, kpss_p_full=0.10, break_p=0.40)
    assert st.verdict(real)["signal"] == "Real"
    assert st.verdict({**real, "break_p": 0.01, "adf_p_pre": 0.2})["signal"] == "Weak"
    assert st.verdict({**real, "kpss_p_full": 0.01})["signal"] != "Real"
    assert st.verdict({**real, "t_gross": 1.9})["signal"] != "Real"


def test_verdict_mixed_weak_none():
    assert st.verdict(_h(adf_p_pre=0.01))["signal"] == "Mixed"
    assert st.verdict(_h())["signal"] == "Weak"
    assert st.verdict(_h(t_gross=0.5, adf_p_full=0.03))["signal"] == "Weak"
    assert st.verdict(_h(t_gross=0.5, break_p=0.30))["signal"] == "None"


def test_verdict_tradability_never_investable_on_spot():
    assert st.verdict(_h())["trad"] == "Mirage"
    assert st.verdict(_h(t_net=2.5, frozen_underwater_months=12))["trad"] == "Fragile"
    assert st.verdict(_h(t_net=2.5, frozen_underwater_months=60))["trad"] == "Mirage"
    assert st.verdict(_h(t_net=9.0, frozen_underwater_months=0))["trad"] != "Investable"


def test_verdict_prose_carries_the_labels():
    v = st.verdict(_h())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "upper bound" in v["trad_why"]
    assert "bootstrap p" in v["signal_why"]
    assert "read in dollars it would have been" in v["signal_why"]
    assert "2010-09" in v["one_sentence"]


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_full_sample_log_ratio_is_not_stationary():
    sf = st.spread_frame(data.load_crude())
    assert st.adf(sf["log_ratio"])["p"] > 0.05
    assert st.kpss_test(sf["log_ratio"])["reject5"]


@needs_real
def test_real_break_lands_in_the_cushing_episode():
    sf = st.spread_frame(data.load_crude())
    b = st.sup_f_bootstrap(sf["log_ratio"], n_boot=99)
    assert pd.Timestamp("2010-01-01") <= b["break_date"] <= pd.Timestamp("2011-12-31")
    assert b["p"] < 0.05


@needs_real
def test_real_pre_2010_dollar_spread_is_tethered():
    sf = st.spread_frame(data.load_crude())
    assert st.adf(sf["spread"][:"2009-12-31"])["p"] < 0.05
    assert st.ar1_fit(sf["log_ratio"][:"2010-08-31"])["halflife"] < 6


@needs_real
def test_real_frozen_trader_goes_short_and_stays_underwater():
    px = data.load_crude()
    sf = st.spread_frame(px)
    bk = st.book(px, st.positions(st.zscore_frozen(sf["log_ratio"])))
    ep = st.episodes(bk)
    assert ep.iloc[0]["side"] == "short spread"
    assert ep.iloc[0]["mae_usd"] < -15
    assert st.underwater(bk["net"], "2010-01-31")["months"] > 36
