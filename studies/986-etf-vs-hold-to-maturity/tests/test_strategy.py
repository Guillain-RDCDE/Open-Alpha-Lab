"""Strategy tests for Study 986 — bond arithmetic first, then the roll."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Bond arithmetic — the textbook identities, as tests
# --------------------------------------------------------------------------- #
def test_a_bond_priced_at_its_coupon_trades_at_par():
    for y in (0.01, 0.04, 0.09):
        for m in (2.0, 10.0, 30.0):
            assert st.price_from_yield(y, m, y) == pytest.approx(100.0, abs=1e-8)


def test_price_falls_when_yield_rises():
    assert st.price_from_yield(0.05, 10, 0.04) < st.price_from_yield(0.03, 10, 0.04)


def test_a_zero_maturity_bond_is_worth_par():
    assert st.price_from_yield(0.05, 0.0, 0.04) == 100.0


def test_duration_of_a_zero_coupon_bond_is_its_maturity():
    """The definition, and a check that the cash-flow weighting is right."""
    d = st.macaulay_duration(0.04, 10.0, coupon=0.0, freq=1)
    assert d == pytest.approx(10.0, abs=1e-9)


def test_duration_of_a_coupon_bond_is_less_than_its_maturity():
    assert st.macaulay_duration(0.04, 10.0) < 10.0
    assert st.macaulay_duration(0.04, 30.0) < 30.0


def test_duration_rises_with_maturity_and_falls_with_yield():
    assert st.macaulay_duration(0.04, 20.0) > st.macaulay_duration(0.04, 10.0)
    assert st.macaulay_duration(0.08, 20.0) < st.macaulay_duration(0.02, 20.0)


def test_modified_duration_predicts_the_price_move():
    y, m = 0.04, 10.0
    d = st.modified_duration(y, m, y)
    dy = 0.0001
    actual = (st.price_from_yield(y + dy, m, y) - st.price_from_yield(y, m, y)) / 100.0
    assert actual == pytest.approx(-d * dy, rel=0.01)


# --------------------------------------------------------------------------- #
# The two instruments
# --------------------------------------------------------------------------- #
def test_a_bond_held_to_maturity_earns_its_yield_on_a_flat_curve():
    """The claim that makes the whole comparison interesting — verified exactly."""
    y, m = 0.04, 10.0
    rates = st.rate_path("flat", n=int(m * 252), start=y)
    b = st.simulate_bond(rates, m)
    total = b["value"].iloc[-1] / b["value"].iloc[0]
    assert total ** (1 / m) - 1 == pytest.approx(y, abs=0.004)


def test_a_bond_pulls_to_par_whatever_rates_do():
    y, m = 0.04, 10.0
    n = int(m * 252)
    for kind in ("flat", "shock", "ramp", "roundtrip"):
        rates = st.rate_path(kind, n=n, start=y, end=0.08)
        b = st.simulate_bond(rates, m)
        price_only = st.price_from_yield(rates[-1], b["remaining_maturity"].iloc[-1], y)
        assert price_only == pytest.approx(100.0, abs=0.5)


def test_the_rolling_fund_never_matures():
    rates = st.rate_path("flat", n=5000, start=0.04)
    f = st.simulate_rolling_fund(rates, 10.0)
    assert f["duration"].iloc[-1] == pytest.approx(f["duration"].iloc[0], rel=0.01)
    assert f["duration"].iloc[-1] > 7.0


def test_a_bonds_duration_falls_to_zero_and_a_funds_does_not():
    """The structural difference, in one assertion."""
    m = 10.0
    n = int(m * 252)
    rates = st.rate_path("flat", n=n, start=0.04)
    c = st.compare(rates, m)
    assert c["remaining_maturity"].iloc[-1] == pytest.approx(0.0, abs=0.01)
    assert c["fund_duration"].iloc[-1] > 7.0


def test_on_a_flat_curve_the_two_are_nearly_identical():
    rates = st.rate_path("flat", n=2520, start=0.04)
    c = st.compare(rates, 10.0)
    gap = (c["fund_value"] - c["bond_value"]).abs().max()
    assert gap < 0.05


def test_after_a_rate_rise_the_fund_falls_behind_first():
    w = st.synthetic_world(n_years=20, shock_bp=200)
    c = w["comparison"]
    just_after = c.iloc[300]
    assert just_after["fund_value"] < just_after["bond_value"]


def test_after_a_rate_rise_the_fund_eventually_catches_up():
    """The reinvestment side of the trade, which the folklore leaves out."""
    rates = st.rate_path("shock", n=int(30 * 252), start=0.04, end=0.06, shock_at=252)
    x = st.crossover_horizon(rates, 10.0)
    assert np.isfinite(x["crossover_years"])
    assert 2.0 < x["crossover_years"] < 10.0


def test_the_crossover_search_stops_at_the_bonds_maturity():
    """Past redemption the comparison depends on a reinvestment assumption, so it stops."""
    rates = st.rate_path("shock", n=int(60 * 252), start=0.04, end=0.06, shock_at=252)
    x = st.crossover_horizon(rates, 30.0)
    assert np.isnan(x["crossover_years"]) or x["crossover_years"] <= 30.0


def test_the_fund_converges_to_its_starting_yield_on_a_trending_path():
    rates = st.rate_path("ramp", n=int(120 * 252), start=0.04, end=0.10)
    c = st.convergence_horizon(rates, 10.0, tol=0.0005)
    assert np.isfinite(c["convergence_years"])
    assert 3.0 < c["convergence_years"] < 25.0


def test_the_convergence_horizon_scales_with_duration():
    """The load-bearing claim: duration is a clock, not just a risk number."""
    rates = st.rate_path("ramp", n=int(120 * 252), start=0.04, end=0.10)
    out = [(st.convergence_horizon(rates, m, tol=0.0005)) for m in (5.0, 10.0, 20.0, 30.0)]
    years = [o["convergence_years"] for o in out]
    assert years == sorted(years)
    ratios = [o["convergence_years"] / o["duration"] for o in out]
    assert all(0.5 < r < 3.0 for r in ratios)


def test_convergence_lands_below_the_leibowitz_bound():
    """Because the fund's duration *shrinks* as yields rise, the price loss is less than D0*dy.

    Leibowitz, Bova & Kogelman (2014) derive 2D-1 from a fixed duration. Letting duration move
    with the yield — which it does, and which this simulation models — brings the crossing in.
    """
    rates = st.rate_path("ramp", n=int(120 * 252), start=0.04, end=0.10)
    for m in (10.0, 20.0, 30.0):
        c = st.convergence_horizon(rates, m, tol=0.0005)
        assert c["convergence_years"] < c["leibowitz_2d_minus_1"]


def test_convergence_is_nan_when_the_path_is_too_short_to_see_it():
    rates = st.rate_path("ramp", n=int(3 * 252), start=0.04, end=0.041)
    assert np.isnan(st.convergence_horizon(rates, 30.0, tol=1e-9)["convergence_years"])


def test_a_rate_fall_reverses_the_sign_of_the_gap():
    up = st.crossover_horizon(st.rate_path("shock", n=6000, start=0.04, end=0.06), 10.0)
    down = st.crossover_horizon(st.rate_path("shock", n=6000, start=0.04, end=0.02), 10.0)
    assert np.sign(up["initial_gap"]) != np.sign(down["initial_gap"])


def test_rate_path_kinds_are_distinct():
    kinds = {k: st.rate_path(k, n=1000, start=0.03, end=0.06) for k in
             ("flat", "shock", "ramp", "roundtrip")}
    assert all(len(v) == 1000 for v in kinds.values())
    assert kinds["flat"].std() == pytest.approx(0.0, abs=1e-12)
    assert kinds["roundtrip"][-1] == pytest.approx(kinds["roundtrip"][0], abs=0.01)
    with pytest.raises(ValueError):
        st.rate_path("nonsense")


def test_rates_never_go_negative_on_a_noisy_path():
    r = st.rate_path("ramp", n=5000, start=0.001, end=0.002, vol=0.05)
    assert (r > 0).all()


# --------------------------------------------------------------------------- #
# The real-tape machinery
# --------------------------------------------------------------------------- #
def _tape(n=4000, drift=0.0, seed=986):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    y = pd.Series(np.maximum(0.04 + drift * np.arange(n) / n
                             + np.cumsum(rng.normal(0, 0.0008, n)), 0.001), index=idx)
    dy = y.diff().fillna(0.0)
    r = y / 252 - 7.5 * dy
    px = pd.Series(100 * np.cumprod(1 + r), index=idx)
    return px, y


def test_realised_vs_promised_measures_annualised_returns():
    px, y = _tape()
    t = st.realised_vs_promised(px, y, horizon_y=2.0, duration=7.5)
    assert len(t) > 20
    i = 0
    step = int(2.0 * 252)
    expected = (px.iloc[i + step] / px.iloc[i]) ** 0.5 - 1
    assert t["realised"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_realised_vs_promised_is_empty_when_the_horizon_exceeds_the_data():
    px, y = _tape(n=300)
    assert len(st.realised_vs_promised(px, y, 5.0, 7.5)) == 0


def test_the_error_is_explained_by_duration_times_the_rate_change():
    """The mechanism test: the shortfall is the roll, not something else."""
    px, y = _tape(n=6000)
    t = st.realised_vs_promised(px, y, horizon_y=3.0, duration=7.5)
    d = st.error_decomposition(t, duration=7.5, horizon_y=3.0)
    assert d["slope"] == pytest.approx(1.0, abs=0.25)
    assert d["r2"] > 0.8


def test_error_decomposition_declines_on_too_few_windows():
    px, y = _tape(n=800)
    t = st.realised_vs_promised(px, y, 2.0, 7.5)
    assert "slope" not in st.error_decomposition(t, 7.5, 2.0) or len(t) >= 30


def test_the_error_shrinks_with_horizon_but_does_not_vanish():
    px, y = _tape(n=8000)
    c = st.convergence_by_horizon(px, y, duration=7.5, horizons=(1, 3, 7))
    assert c["sd_error"].iloc[0] > c["sd_error"].iloc[-1]
    assert c["sd_error"].iloc[-1] > 0


def test_convergence_table_is_empty_on_a_short_tape():
    px, y = _tape(n=400)
    assert len(st.convergence_by_horizon(px, y, 7.5, horizons=(5, 10))) == 0


def test_fund_durations_are_declared_and_ordered():
    d = st.FUND_DURATION
    assert d["SHY"] < d["IEF"] < d["TLT"]
    assert d["BIL"] < d["SHY"]


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_windows": 180, "duration": 7.5, "fund": "IEF",
         "mean_error_at_duration": -0.004, "sd_error_at_duration": 0.016,
         "share_within_1pp": 0.41, "decomp_slope": 0.94, "decomp_r2": 0.86,
         "sd_error_1y": 0.061, "sd_error_10y": 0.011, "crossover_years": 5.6,
         "convergence_years": 11.1, "leibowitz_bound": 15.4,
         "initial_gap": -0.11, "sim_maturity": 10.0, "sim_shock_bp": 200.0,
         "sim_duration": 8.1}
    h.update(over)
    return h


def test_verdict_signal_needs_both_the_phenomenon_and_the_mechanism():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(decomp_r2=0.2))["signal"] == "Partial"
    assert st.verdict(_headline(sd_error_at_duration=0.002))["signal"] == "Partial"
    assert st.verdict(_headline(sd_error_at_duration=0.002, decomp_r2=0.2))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(convergence_years=40))["trad"] == "Partial"
    assert st.verdict(_headline(convergence_years=40, sd_error_1y=0.001))["trad"] == "Mirage"


def test_verdict_tolerates_a_convergence_that_never_happens():
    v = st.verdict(_headline(convergence_years=float("nan")))
    assert v["trad"] in {"Partial", "Mirage"}


def test_verdict_prose_names_the_leibowitz_bound():
    v = st.verdict(_headline())
    assert "Leibowitz" in v["trad_why"] and "2D" in v["trad_why"]


def test_verdict_prose_states_the_duration_result():
    v = st.verdict(_headline())
    assert "duration" in v["one_sentence"]
    assert "immunisation" in v["signal_why"] or "duration" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
