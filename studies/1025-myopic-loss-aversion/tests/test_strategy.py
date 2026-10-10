"""Strategy tests for Study 1025 — prospect theory, the break-even horizon, the switcher."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dontlook import data, strategy as st  # noqa: E402

needs_monthly = pytest.mark.skipif(not data.have_monthly(),
                                   reason="arch tapes not installed")
H = st.HORIZONS_M


# --------------------------------------------------------------------------- #
# Prospect theory
# --------------------------------------------------------------------------- #
def test_weighting_function_endpoints_and_inverse_s():
    for g in (0.61, 0.69):
        assert st.tk_weight(0.0, g) == pytest.approx(0.0)
        assert st.tk_weight(1.0, g) == pytest.approx(1.0)
        assert st.tk_weight(0.01, g) > 0.01        # small probabilities overweighted
        assert st.tk_weight(0.9, g) < 0.9          # large ones underweighted
    p = np.linspace(0, 1, 11)
    assert np.allclose(st.tk_weight(p, 1.0), p)


def test_decision_weights_sum_to_one_on_one_sided_prospects():
    wl, wg = st.decision_weights(250)
    assert wl.sum() == pytest.approx(1.0)
    assert wg.sum() == pytest.approx(1.0)
    wl0, wg0 = st.decision_weights(250, weighting=False)
    assert np.allclose(wl0, 1 / 250) and np.allclose(wg0, 1 / 250)


def test_cpt_without_weighting_or_loss_aversion_is_the_mean():
    x = np.random.default_rng(0).normal(0.01, 0.05, 1000)
    v = st.cpt_value(x, alpha=1.0, lam=1.0, weighting=False)
    assert v == pytest.approx(x.mean())


def test_a_fair_coin_flip_is_unattractive_to_a_loss_averse_investor():
    x = np.array([-0.1, 0.1] * 50)
    assert st.cpt_value(x) < 0
    assert st.cpt_value(x, lam=1.0, weighting=False) == pytest.approx(0.0)


def test_value_is_homogeneous_so_units_do_not_matter():
    x = np.random.default_rng(1).normal(0.005, 0.05, 500)
    assert st.cpt_value(100 * x) == pytest.approx(100 ** st.ALPHA * st.cpt_value(x))


def test_cpt_vectorises_over_rows():
    X = np.random.default_rng(2).normal(0.0, 0.05, (4, 300))
    rows = [st.cpt_value(r) for r in X]
    assert np.allclose(st.cpt_value(X), rows)


def test_more_loss_aversion_lowers_the_value_of_a_risky_prospect():
    x = np.random.default_rng(3).normal(0.01, 0.05, 1000)
    assert st.cpt_value(x, lam=3.0) < st.cpt_value(x, lam=2.25) < st.cpt_value(x, lam=1.0)


# --------------------------------------------------------------------------- #
# Horizons
# --------------------------------------------------------------------------- #
def test_horizon_returns_compound_correctly():
    r = np.array([0.1, -0.1, 0.2, 0.0])
    ov = st.horizon_returns(r, 2)
    assert np.allclose(ov, [1.1 * 0.9 - 1, 0.9 * 1.2 - 1, 1.2 - 1])
    no = st.horizon_returns(r, 2, overlapping=False)
    assert np.allclose(no, [1.1 * 0.9 - 1, 1.2 - 1])
    assert st.horizon_returns(r, 9).size == 0


def test_loss_probability_falls_with_horizon_when_a_premium_is_planted(planted):
    L = st.loss_probability_curve(planted, (1, 12, 120))
    assert L.loc[1, "stock"] > L.loc[12, "stock"] > L.loc[120, "stock"]
    assert L.loc[1, "stock"] > 0.4 and L.loc[120, "stock"] < 0.1
    assert (L["bill"] == 0).all()


def test_loss_probability_does_not_fall_to_zero_under_the_null(null_world):
    L = st.loss_probability_curve(null_world, (1, 120))
    assert L.loc[120, "stock"] > 0.25


# --------------------------------------------------------------------------- #
# The break-even
# --------------------------------------------------------------------------- #
def test_break_even_reads_the_last_crossing():
    h = [1, 2, 4, 8]
    assert st.break_even(h, [-1, -1, -1, -1]) == np.inf
    assert st.break_even(h, [1, 1, 1, 1]) == 1.0
    assert st.break_even(h, [-1, -1, 1, 1]) == pytest.approx(np.sqrt(8))  # log-midpoint 2..4
    # a positive blip that falls back does not count
    assert st.break_even(h, [-1, 1, -1, 1]) == pytest.approx(np.sqrt(32))


def test_planted_world_has_a_break_even_near_a_year(planted):
    P = st.pt_curve(planted, H)
    be_b = st.break_even(H, (P["stock"] - P["bond"]).to_numpy())
    be_f = st.break_even(H, (P["stock"] - P["bill"]).to_numpy())
    assert 4 < be_b < 30
    assert 6 < be_f < 36


def test_null_world_has_no_break_even_against_bills(null_world):
    """At signal_strength 0 no horizon on the grid makes stocks attractive."""
    P = st.pt_curve(null_world, H)
    assert st.break_even(H, (P["stock"] - P["bill"]).to_numpy()) == np.inf
    assert st.break_even(H, (P["stock"] - P["bond"]).to_numpy()) == np.inf
    assert ((P["stock"] - P["bill"]) < 0).all()


def test_iid_estimator_agrees_with_overlapping_on_an_iid_world(planted):
    Po = st.pt_curve(planted, H)
    Pi = st.pt_curve(planted, H, method="iid", n_iid=20000)
    bo = st.break_even(H, (Po["stock"] - Po["bill"]).to_numpy())
    bi = st.break_even(H, (Pi["stock"] - Pi["bill"]).to_numpy())
    assert bi == pytest.approx(bo, rel=0.35)


def test_nonoverlapping_estimator_runs(planted):
    P = st.pt_curve(planted, (1, 12, 60), method="nonoverlapping")
    assert P.notna().all().all()
    with pytest.raises(ValueError):
        st.pt_curve(planted, (1,), method="bogus")


def test_bootstrap_is_reproducible_and_shaped(planted):
    df = planted.iloc[:1200]
    a = st.bootstrap_curves(df, (1, 12, 60), n_boot=30, seed=5)
    b = st.bootstrap_curves(df, (1, 12, 60), n_boot=30, seed=5)
    assert a["pt"].shape == (30, 3, 3) and a["loss"].shape == (30, 3, 3)
    assert np.allclose(a["pt"], b["pt"])


def test_cbb_indices_wrap_and_keep_blocks():
    idx = st.cbb_indices(10, 4, 3, np.random.default_rng(0))
    assert idx.shape == (3, 10)
    d = np.diff(idx[:, :4], axis=1) % 10
    assert (d == 1).all()


def test_bootstrap_flags_the_short_horizon_gap_as_significant_on_a_long_planted_world(planted):
    P = st.pt_curve(planted, H)
    boot = st.bootstrap_curves(planted, H, n_boot=80, seed=1)
    s = st.summarise_gap(P, boot, "stock", "bill")
    assert s["table"].loc[1, "gap"] < 0 and s["table"].loc[1, "p"] < 0.05
    assert s["table"].loc[120, "gap"] > 0 and s["table"].loc[120, "p"] < 0.05
    assert np.isfinite(s["be_hi"]) and s["be_lo"] <= s["break_even"] <= s["be_hi"]


def test_two_sided_p():
    assert st.two_sided_p(np.ones(100)) == 0.0
    assert st.two_sided_p(np.r_[np.ones(50), -np.ones(50)]) == 1.0


def test_param_sweep_without_loss_aversion_is_censored_at_one_month(planted):
    sw = st.param_sweep(planted.iloc[:2400], "stock", "bill", horizons=H,
                        alphas=(0.88,), lambdas=(1.0, 2.25, 3.0), weighting=False)
    assert sw.loc[1.0, 0.88] == 1.0
    assert sw.loc[3.0, 0.88] >= sw.loc[2.25, 0.88] > 1.0


def test_break_even_lambda_makes_the_gap_vanish(planted):
    lam = st.break_even_lambda(planted, 12, b="bill")
    Ra = st.horizon_returns(planted["stock"].to_numpy(), 12)
    Rb = st.horizon_returns(planted["bill"].to_numpy(), 12)
    gap = st.cpt_value(Ra, lam=lam) - st.cpt_value(Rb, lam=lam)
    assert abs(gap) < 1e-6 and 1.0 < lam < 5.0


# --------------------------------------------------------------------------- #
# The myopic switcher
# --------------------------------------------------------------------------- #
def _toy():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    s = pd.Series([0.05, -0.02, 0.03, -0.01, -0.04, 0.02], index=idx)
    b = pd.Series(0.001, index=idx)
    return s, b


def test_switcher_has_exactly_one_lag():
    s, b = _toy()
    sw = st.myopic_switcher(s, b, "M", cost_bps=0.0)
    # month 1 held (start invested); then the sign of month t sets month t+1
    assert sw["w"].tolist() == [1, 1, 0, 1, 0, 0]
    assert np.allclose(sw["gross"], np.where(sw["w"] == 1, s, b))


def test_switcher_does_not_look_ahead():
    s, b = _toy()
    w0 = st.myopic_switcher(s, b, "M")["w"]
    s2 = s.copy()
    s2.iloc[3] = -0.5                              # change month 4's return …
    w1 = st.myopic_switcher(s2, b, "M")["w"]
    assert w0.iloc[:4].tolist() == w1.iloc[:4].tolist()   # … months 1-4 positions unchanged


def test_switcher_charges_one_way_cost_per_switch():
    s, b = _toy()
    sw = st.myopic_switcher(s, b, "M", cost_bps=10.0)
    assert sw["turnover"].tolist() == [0, 0, 1, 1, 1, 0]
    assert np.allclose(sw["gross"] - sw["net"], sw["turnover"] * 0.001)


def test_switcher_aggregates_daily_returns_into_periods():
    idx = pd.bdate_range("2021-01-04", periods=10)            # two Mon-Fri weeks
    s = pd.Series([0.01] * 4 + [-0.06] + [0.01] * 5, index=idx)
    b = pd.Series(0.0, index=idx)
    sw = st.myopic_switcher(s, b, "W", cost_bps=0.0)
    assert (sw["w"].iloc[:5] == 1).all() and (sw["w"].iloc[5:] == 0).all()
    swd = st.myopic_switcher(s, b, "D", cost_bps=0.0)
    assert swd["w"].iloc[5] == 0 and swd["w"].iloc[6] == 1


def test_period_codes():
    idx = pd.to_datetime(["2020-04-30", "2020-05-29", "2020-12-31", "2021-01-29"])
    assert len(set(st.period_codes(idx, "Q"))) == 3
    assert len(set(st.period_codes(idx, "Y"))) == 2
    with pytest.raises(ValueError):
        st.period_codes(idx, "X")


def test_myopia_is_costly_when_a_premium_is_planted(planted):
    t = st.discipline_table(planted, ["M", "Y"], 12, cost_bps=5.0, n_boot=200)
    assert (t["d_cagr_net"] > 0).all()
    assert t.loc["M", "p_ret"] < 0.05 and t.loc["M", "t_hac"] > 2


def test_myopia_is_free_gross_when_there_is_no_premium(null_world):
    sw = st.myopic_switcher(null_world["stock"], null_world["bill"], "M", cost_bps=0.0)
    r = st.discipline_test(sw, 12, block=12, n_boot=200)
    assert abs(r["t_hac"]) < 2 and r["p_ret"] > 0.05


def test_daily_myopia_burns_money_on_costs(planted_daily):
    t = st.discipline_table(planted_daily, ["D", "M"], 252, cost_bps=5.0, n_boot=100,
                            base_block=21)
    assert t.loc["D", "switches_per_year"] > 80
    assert t.loc["D", "d_cagr_net"] > t.loc["M", "d_cagr_net"] > 0


def test_cost_sweep_is_monotone(planted_daily):
    cs = st.cost_sweep(planted_daily, "W", 252, costs=(0.0, 5.0, 25.0))
    assert cs["d_cagr"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _disc(d_cagr=(0.11, 0.08, 0.01, 0.01, 0.01), p=(0.0, 0.0, 0.2, 0.2, 0.2),
          d_sh=(0.9, 0.5, -0.05, -0.04, -0.03), p_sh=(0.0, 0.0, 0.6, 0.7, 0.9)):
    out = []
    for f, a, b, c, e in zip(("D", "W", "M", "Q", "Y"), d_cagr, p, d_sh, p_sh):
        out.append({"freq": f, "d_cagr_net": a, "p_ret": b, "d_sharpe": c, "p_sharpe": e,
                    "t_hac": 3.0 if b < 0.05 else 1.2,
                    "switches_per_year": {"D": 129, "W": 27, "M": 5, "Q": 1.6, "Y": 0.3}[f]})
    return out


def _h(**over):
    h = {"gap_short": -0.023, "p_short": 0.0, "gap_long": 0.56, "p_long": 0.04,
         "be": 30.0, "be_lo": 2.6, "be_hi": 113.0, "be_bills": 17.0, "be_bills_lo": 4.2,
         "be_bills_hi": 46.0, "be_pre": 11.0, "be_post": 43.0, "be_real": 16.0,
         "be_iid": 11.6, "be_bt": 23.0, "lambda_12": 1.79, "loss_1d": 0.465,
         "loss_1m": 0.37, "loss_12m": 0.25, "loss_120m": 0.05, "discipline": _disc()}
    h.update(over)
    return h


def test_verdict_signal_all_four_branches():
    assert st.verdict(_h())["signal"] == "Weak"
    assert st.verdict(_h(be_lo=6.0, be_hi=24.0))["signal"] == "Real"
    assert st.verdict(_h(be_lo=6.0, be_hi=24.0, be_post=np.inf))["signal"] == "Mixed"
    assert st.verdict(_h(p_long=0.2))["signal"] == "None"
    assert st.verdict(_h(gap_short=0.01))["signal"] == "None"


def test_verdict_pinning_requires_the_interval_to_contain_a_year():
    assert st.verdict(_h(be_lo=14.0, be_hi=30.0))["signal"] == "Weak"
    assert st.verdict(_h(be_lo=6.0, be_hi=np.inf))["signal"] == "Weak"


def test_verdict_tradability_all_three_branches():
    assert st.verdict(_h())["trad"] == "Fragile"
    assert st.verdict(_h(discipline=_disc(p=(0.2,) * 5)))["trad"] == "Mirage"
    assert st.verdict(_h(discipline=_disc(d_sh=(0.5,) * 5, p_sh=(0.01,) * 5)))["trad"] \
        == "Investable"
    # a significant gap in the WRONG direction (switcher wins) is not a cost of myopia
    assert st.verdict(_h(discipline=_disc(d_cagr=(-0.02,) * 5, p=(0.01,) * 5)))["trad"] \
        == "Mirage"


def test_verdict_prose_is_complete_and_honest():
    v = st.verdict(_h())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "i.i.d." in v["signal_why"] and "too wide" in v["signal_why"]
    assert "seatbelt" in v["trad_why"] and "daily and weekly" in v["one_sentence"]
    v2 = st.verdict(_h(p_long=0.3))
    assert "does not flip" in v2["signal_why"]


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_monthly
def test_real_tape_point_estimates():
    m = data.load_monthly()[["stock", "bond", "bill"]]
    L = st.loss_probability_curve(m, (1, 12, 120))
    assert 0.30 < L.loc[1, "stock"] < 0.45
    assert L.loc[120, "stock"] < L.loc[12, "stock"] < L.loc[1, "stock"]
    P = st.pt_curve(m, H)
    be = st.break_even(H, (P["stock"] - P["bond"]).to_numpy())
    assert 6 < be < 60
    Pi = st.pt_curve(m, H, method="iid", n_iid=10000)
    assert st.break_even(H, (Pi["stock"] - Pi["bond"]).to_numpy()) < be
