"""Strategy tests for Study 1011 — decay, breadth, and the optimal trading rate."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from halflife import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def test_zscores_are_centred_and_scaled_row_by_row():
    df = pd.DataFrame(np.random.default_rng(0).normal(5, 3, (50, 10)))
    z = st.zscore_cross_section(df)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1, ddof=0), 1.0, atol=1e-9)


def test_a_flat_cross_section_does_not_blow_up():
    df = pd.DataFrame(np.ones((10, 5)))
    assert st.zscore_cross_section(df).isna().all().all()


def test_reversal_buys_the_losers():
    idx = pd.bdate_range("2020-01-01", periods=30)
    R = pd.DataFrame({"UP": np.full(30, 0.01), "DOWN": np.full(30, -0.01)}, index=idx)
    s = st.reversal_signal(R, 5).dropna()
    assert s["DOWN"].iloc[-1] > s["UP"].iloc[-1]


def test_momentum_buys_the_winners_and_skips_the_last_month():
    idx = pd.bdate_range("2018-01-01", periods=400)
    R = pd.DataFrame({"UP": np.full(400, 0.001), "DOWN": np.full(400, -0.001)},
                     index=idx)
    s = st.momentum_signal(R, 252, 21).dropna()
    assert s["UP"].iloc[-1] > s["DOWN"].iloc[-1]


def test_low_vol_prefers_the_calm_name():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    R = pd.DataFrame({"CALM": rng.normal(0, 0.002, 200),
                      "WILD": rng.normal(0, 0.03, 200)}, index=idx)
    s = st.volatility_signal(R, 63).dropna()
    assert s["CALM"].iloc[-1] > s["WILD"].iloc[-1]


def test_make_signals_spans_the_decay_spectrum():
    px = data.load_prices()
    R = _panel(px)
    sigs = st.make_signals(R)
    assert set(sigs) == {"reversal_5d", "reversal_21d", "momentum_12m", "low_vol_63d"}


# --------------------------------------------------------------------------- #
# Forward returns and IC
# --------------------------------------------------------------------------- #
def test_forward_returns_look_forward():
    idx = pd.bdate_range("2020-01-01", periods=50)
    R = pd.DataFrame({"A": np.arange(50) * 0.001}, index=idx)
    f = st.forward_returns(R, 5)
    expected = np.log1p(R["A"].iloc[1:6]).sum()
    assert f["A"].iloc[0] == pytest.approx(expected)


def test_forward_returns_are_nan_at_the_end():
    R = pd.DataFrame({"A": np.full(50, 0.001)},
                     index=pd.bdate_range("2020-01-01", periods=50))
    assert st.forward_returns(R, 5)["A"].iloc[-1] != st.forward_returns(R, 5)["A"].iloc[-1]


def test_ic_recovers_a_planted_relationship():
    w = st.synthetic_panel(n_assets=50, n_days=4000, half_life=21, ic=0.10)
    d = st.ic_at_horizon(w["signal"], w["returns"], 1)
    assert d["ic"] > 0.04
    assert d["t"] > 3


def test_ic_is_near_zero_when_nothing_is_planted():
    w = st.synthetic_panel(n_assets=50, n_days=4000, half_life=21, ic=0.0)
    d = st.ic_at_horizon(w["signal"], w["returns"], 1)
    assert abs(d["ic"]) < 0.02


def test_ic_declines_on_too_few_dates():
    """Fewer than 30 usable cross-sections is not a measurement."""
    w = st.synthetic_panel(n_assets=50, n_days=40)
    assert st.ic_at_horizon(w["signal"], w["returns"], 21) == {}


def test_ic_declines_when_the_cross_section_is_too_thin():
    w = st.synthetic_panel(n_assets=4, n_days=2000)
    assert st.ic_at_horizon(w["signal"], w["returns"], 21) == {}


def test_the_vectorised_ic_matches_a_naive_loop():
    """The optimisation is only worth having if it computes the same thing."""
    w = st.synthetic_panel(n_assets=30, n_days=600, half_life=10, ic=0.10)
    s, r = w["signal"], w["returns"]
    fwd = st.forward_returns(r, 5)
    common = s.index.intersection(fwd.index)
    naive = []
    for d in common:
        a, b = s.loc[d], fwd.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() >= 10:
            naive.append(float(a[ok].rank().corr(b[ok].rank())))
    naive = np.array([x for x in naive if np.isfinite(x)])
    assert st.ic_at_horizon(s, r, 5)["ic"] == pytest.approx(float(naive.mean()),
                                                            abs=1e-9)


# --------------------------------------------------------------------------- #
# Decay
# --------------------------------------------------------------------------- #
def test_the_cumulative_ic_can_rise_even_while_the_signal_decays():
    """Which is why the half-life is fitted to the MARGINAL profile."""
    w = st.synthetic_panel(n_assets=50, n_days=6000, half_life=10, ic=0.08)
    d = st.ic_decay(w["signal"], w["returns"], horizons=(1, 5, 21, 63))
    assert d["ic"].iloc[-1] > d["ic"].iloc[0]
    m = st.marginal_ic(d)
    assert m["marginal_ic_per_day"].iloc[-1] < m["marginal_ic_per_day"].iloc[0]


def test_the_fitted_half_life_recovers_the_planted_one():
    """The calibration the whole study rests on."""
    for true_hl in (5.0, 21.0, 63.0):
        w = st.synthetic_panel(n_assets=60, n_days=8000, half_life=true_hl, ic=0.10)
        f = st.fit_half_life(st.lag_profile(w["signal"], w["returns"]))
        assert f["decaying"]
        assert 0.5 * true_hl < f["half_life"] < 2.0 * true_hl


def test_the_estimator_cannot_resolve_a_half_life_of_a_couple_of_days():
    """A real limit, stated rather than hidden.

    With a daily lag grid there are only two or three usable points before the IC reaches the
    noise floor, so a very fast signal's half-life is overstated. Anything the estimator
    reports below about five days should be read as "fast", not as a number.
    """
    w = st.synthetic_panel(n_assets=60, n_days=8000, half_life=2.0, ic=0.10)
    f = st.fit_half_life(st.lag_profile(w["signal"], w["returns"]))
    assert f["half_life"] > 2.0 * 2.0


def test_fitting_the_CUMULATIVE_profile_understates_the_half_life_badly():
    """The mistake this study made first, pinned so it cannot come back.

    Differencing a cumulative IC mixes the signal's decay with the sqrt(h) growth of the
    cumulative return's standard deviation, and the fitted half-life comes out several times
    too short.
    """
    w = st.synthetic_panel(n_assets=60, n_days=8000, half_life=63.0, ic=0.10)
    good = st.fit_half_life(st.lag_profile(w["signal"], w["returns"]))
    bad = st.fit_half_life(st.marginal_ic(st.ic_decay(
        w["signal"], w["returns"], (1, 2, 3, 5, 8, 13, 21, 34, 55, 89)))
        .rename(columns={"marginal_ic_per_day": "ic"}))
    assert good["half_life"] > 2 * bad["half_life"]


def test_a_faster_planted_signal_fits_a_shorter_half_life():
    fast = st.synthetic_panel(n_assets=60, n_days=8000, half_life=5, ic=0.10)
    slow = st.synthetic_panel(n_assets=60, n_days=8000, half_life=63, ic=0.10)
    hf = st.fit_half_life(st.lag_profile(fast["signal"], fast["returns"]))
    hs = st.fit_half_life(st.lag_profile(slow["signal"], slow["returns"]))
    assert hf["half_life"] < hs["half_life"]


def test_the_lag_profile_decays_where_the_cumulative_one_climbs():
    w = st.synthetic_panel(n_assets=60, n_days=6000, half_life=21, ic=0.10)
    lag = st.lag_profile(w["signal"], w["returns"], lags=(1, 5, 21, 63))
    cum = st.ic_decay(w["signal"], w["returns"], horizons=(1, 5, 21, 63))
    assert lag["ic"].is_monotonic_decreasing
    assert cum["ic"].iloc[-1] > cum["ic"].iloc[0]


def test_a_non_decaying_signal_is_reported_as_such():
    f = st.fit_half_life(pd.DataFrame(
        {"ic": [0.01, 0.03, 0.06, 0.11, 0.20]},
        index=pd.Index([1, 3, 5, 10, 21], name="lag")))
    assert not f["decaying"]
    assert f["half_life"] == np.inf


def test_fit_declines_on_too_few_points():
    assert st.fit_half_life(pd.DataFrame({"ic": [0.05, 0.04]}, index=[1, 5])) == {}
    assert st.marginal_ic(pd.DataFrame({"ic": [0.05]}, index=[1])).empty


def test_real_signals_have_different_half_lives():
    px = data.load_prices()
    R = _panel(px)
    sigs = st.make_signals(R)
    hls = {}
    for name, s in sigs.items():
        f = st.fit_half_life(st.lag_profile(s, R, lags=(1, 3, 5, 10, 21, 42, 63, 126)))
        if f and f.get("decaying"):
            hls[name] = f["half_life"]
    assert len(hls) >= 2
    assert max(hls.values()) / min(hls.values()) > 1.5


def test_the_half_life_is_estimated_with_real_uncertainty():
    """The study's own contribution — and the reason not to over-tune the trading rate."""
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    u = st.decay_uncertainty(s, R, n_boot=25)
    assert u
    assert u["ratio_95_05"] > 1.5


def test_decay_uncertainty_declines_on_a_short_panel():
    w = st.synthetic_panel(n_assets=20, n_days=400)
    assert st.decay_uncertainty(w["signal"], w["returns"]) == {}


# --------------------------------------------------------------------------- #
# Breadth
# --------------------------------------------------------------------------- #
def test_breadth_counts_time_bets_from_the_half_life():
    b = st.effective_breadth(50, 21)
    assert b["time_bets"] == pytest.approx(252 / 21)
    assert b["breadth"] == pytest.approx(50 * 252 / 21)


def test_a_slower_signal_has_less_breadth():
    assert st.effective_breadth(50, 63)["breadth"] < st.effective_breadth(50, 5)["breadth"]


def test_correlated_bets_reduce_breadth():
    """Buckle's correction, and the part everyone omits."""
    indep = st.effective_breadth(50, 21, correlation=0.0)
    corr = st.effective_breadth(50, 21, correlation=0.3)
    assert corr["breadth"] < indep["breadth"]
    assert corr["cross_sectional_bets"] < 5


def test_the_naive_count_overstates_breadth_enormously():
    b = st.effective_breadth(50, 21, correlation=0.25)
    assert b["overstatement"] > 20


def test_grinold_ir_matches_its_formula():
    assert st.grinold_ir(0.05, 400) == pytest.approx(0.05 * 20)
    assert st.grinold_ir(0.05, 400, transfer=0.5) == pytest.approx(0.5)


def test_residual_correlation_is_much_lower_than_raw_correlation():
    px = data.load_prices()
    R = _panel(px)
    raw = R.corr().to_numpy()
    raw_off = float(np.nanmean(raw[~np.eye(len(raw), dtype=bool)]))
    assert st._residual_correlation(R) < raw_off


def test_residual_correlation_declines_gracefully():
    assert st._residual_correlation(pd.DataFrame({"A": [0.01, 0.02]})) == 0.0


def test_grinold_predicts_the_right_order_of_magnitude():
    """An approximation, checked as one — the gap is the finding, not a failure."""
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    g = st.grinold_check(s, R, half_life=5, rebalance=5)
    assert g
    assert g["predicted_ir_naive"] > g["predicted_ir"]


# --------------------------------------------------------------------------- #
# Trading rate
# --------------------------------------------------------------------------- #
def test_the_backtest_is_dollar_neutral_and_unit_gross():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    b = st.backtest(s, R, rebalance=21, cost_bps=0.0)
    assert b["vol"] > 0
    assert np.isfinite(b["ir"])


def test_costs_reduce_the_information_ratio():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    free = st.backtest(s, R, 5, cost_bps=0.0)
    paid = st.backtest(s, R, 5, cost_bps=50.0)
    assert paid["ir"] < free["ir"]
    assert paid["cost_drag"] > 0


def test_partial_trading_reduces_turnover():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    full = st.backtest(s, R, 21, 10.0, trade_rate=1.0)
    part = st.backtest(s, R, 21, 10.0, trade_rate=0.2)
    assert part["turnover_pa"] < full["turnover_pa"]


def test_faster_rebalancing_costs_more():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    fast = st.backtest(s, R, 1, 10.0)
    slow = st.backtest(s, R, 63, 10.0)
    assert fast["turnover_pa"] > slow["turnover_pa"]


def test_the_gp_rate_rises_with_the_decay_rate():
    """Fast-decaying signals must be traded fast. The unambiguous half of the theory."""
    fast = st.gp_trade_rate(2.0, cost_bps=10.0)
    slow = st.gp_trade_rate(120.0, cost_bps=10.0)
    assert fast > slow


def test_the_gp_rate_falls_as_costs_rise():
    cheap = st.gp_trade_rate(21.0, cost_bps=1.0)
    dear = st.gp_trade_rate(21.0, cost_bps=200.0)
    assert dear < cheap


def test_the_gp_rate_stays_a_fraction():
    for hl in (1.0, 5.0, 21.0, 252.0):
        for c in (0.1, 10.0, 500.0):
            assert 0.0 <= st.gp_trade_rate(hl, c) <= 1.0


def test_the_trade_rate_sweep_covers_the_range():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    d = st.trade_rate_sweep(s, R, 21, 10.0, rates=(0.1, 0.5, 1.0))
    assert list(d.index) == [0.1, 0.5, 1.0]
    assert d["turnover_pa"].is_monotonic_increasing


def test_the_rebalance_sweep_reports_against_the_half_life():
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    d = st.rebalance_sweep(s, R, half_life=5.0, periods=(1, 5, 21))
    assert "vs_half_life" in d.columns
    assert d.loc[5, "vs_half_life"] == pytest.approx(1.0)


def test_holding_far_longer_than_the_half_life_wastes_the_signal():
    """The framework's central practical prediction, on a fast signal."""
    px = data.load_prices()
    R = _panel(px)
    s = st.reversal_signal(R, 5)
    d = st.rebalance_sweep(s, R, half_life=5.0, periods=(5, 126), cost_bps=0.0)
    assert d.loc[5, "ir"] > d.loc[126, "ir"]


def test_the_backtest_declines_on_a_degenerate_signal():
    px = data.load_prices()
    R = _panel(px)
    flat = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    assert st.backtest(flat, R, 21) == {}


# --------------------------------------------------------------------------- #
# The control
# --------------------------------------------------------------------------- #
def test_the_synthetic_signal_has_the_persistence_it_claims():
    w = st.synthetic_panel(n_assets=20, n_days=20000, half_life=21)
    s = w["signal"].iloc[:, 0]
    expected_phi = 0.5 ** (1 / 21)
    assert s.autocorr(1) == pytest.approx(expected_phi, abs=0.03)


def test_the_synthetic_ic_is_what_it_claims():
    for target in (0.03, 0.10):
        w = st.synthetic_panel(n_assets=60, n_days=12000, half_life=21, ic=target)
        d = st.ic_at_horizon(w["signal"], w["returns"], 1)
        assert d["ic"] == pytest.approx(target, abs=0.03)


def test_grinold_holds_in_the_world_it_assumes():
    """Independent bets, no costs, no constraints. If it fails here it is not an approximation."""
    w = st.synthetic_panel(n_assets=60, n_days=12000, half_life=21, ic=0.06)
    d = st.ic_at_horizon(w["signal"], w["returns"], 21)
    b = st.effective_breadth(60, 21)
    pred = st.grinold_ir(d["ic"], b["breadth"])
    bt = st.backtest(w["signal"], w["returns"], 21, 0.0)
    assert pred > 0
    assert bt["ir"] > 0
    assert 0.2 < bt["ir"] / pred < 5.0


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _panel(px):
    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2500]
    return px[cols].pct_change().dropna()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_assets": 50, "fastest_hl": 4.0, "fastest_name": "reversal_5d",
         "slowest_hl": 71.0, "slowest_name": "momentum_12m", "hl_spread": 17.8,
         "hl_p05": 3.0, "hl_p95": 12.0, "hl_interval_ratio": 4.0,
         "headline_signal": "reversal_5d", "headline_hl": 4.0, "cost_bps": 10.0,
         "best_rebal": 5, "best_ir": 0.81, "fast_rebal": 5, "fast_ir": 0.42,
         "slow_rebal": 12, "slow_ir": 0.33, "gp_rate": 0.62,
         "best_trade_rate": 0.70, "naive_breadth": 12600.0, "breadth": 218.0,
         "predicted_ir_naive": 3.9, "predicted_ir": 0.51, "realised_ir": 0.81,
         "residual_correlation": 0.14, "breadth_overstatement": 58.0,
         "beats_faster": True, "beats_slower": True}
    h.update(over)
    return h


def test_verdict_signal_needs_a_spread_AND_estimability():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(hl_interval_ratio=25))["signal"] == "Weak"
    assert st.verdict(_headline(hl_spread=1.1))["signal"] == "None"


def test_verdict_tradability_needs_to_beat_both_sides():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(beats_slower=False))["trad"] == "Partial"
    assert st.verdict(_headline(beats_faster=False,
                                beats_slower=False))["trad"] == "Mirage"


def test_verdict_prose_reports_the_breadth_overstatement():
    v = st.verdict(_headline())
    assert "marginal" in v["signal_why"]
    assert "three decimal places" in v["signal_why"]
    assert "overstates breadth" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
