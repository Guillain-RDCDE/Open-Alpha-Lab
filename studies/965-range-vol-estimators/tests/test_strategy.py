"""Strategy tests for Study 965 — the estimators against cases where the answer is known."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from range_vol import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Closed-form cases
# --------------------------------------------------------------------------- #
def test_parkinson_matches_its_formula_on_a_hand_bar():
    bars = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [100.0], "close": [105.0],
                         "volume": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"]))
    expected = np.log(1.10) ** 2 / (4 * np.log(2))
    assert st.parkinson_var(bars).iloc[0] == pytest.approx(expected)


def test_rogers_satchell_is_zero_for_a_monotone_up_day():
    """A day that opens at the low and closes at the high has no RS variance by construction."""
    bars = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [100.0], "close": [110.0],
                         "volume": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"]))
    assert st.rogers_satchell_var(bars).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_estimators_are_zero_on_a_flat_day():
    bars = pd.DataFrame({"open": [100.0] * 3, "high": [100.0] * 3, "low": [100.0] * 3,
                         "close": [100.0] * 3, "volume": [1.0] * 3},
                        index=pd.bdate_range("2020-01-01", periods=3))
    for f in (st.parkinson_var, st.garman_klass_var, st.rogers_satchell_var):
        assert np.allclose(f(bars).to_numpy(), 0.0)
    assert st.close_close_var(bars).iloc[1] == pytest.approx(0.0)


def test_close_close_var_is_the_squared_log_return():
    px = np.array([100.0, 105.0, 99.0])
    bars = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
                         "volume": 1.0}, index=pd.bdate_range("2020-01-01", periods=3))
    assert st.close_close_var(bars).iloc[1] == pytest.approx(np.log(105 / 100) ** 2)


# --------------------------------------------------------------------------- #
# Efficiency where the truth is known
# --------------------------------------------------------------------------- #
def test_range_estimators_beat_close_to_close_when_there_is_no_gap(planted):
    """The textbook claim, in the textbook's world: no overnight gap, known sigma."""
    _, truth = planted
    bars, tr = data.synthetic_ohlc(n_years=12, overnight_share=0.0, signal_strength=1.0,
                                   seed=965)
    tbl = st.efficiency_table(bars, tr["sigma"])
    assert tbl.loc["parkinson", "efficiency_vs_cc"] > 3.0
    assert tbl.loc["garman_klass", "efficiency_vs_cc"] > 3.0
    assert tbl.loc["close_close", "efficiency_vs_cc"] == pytest.approx(1.0)


def test_gap_blind_estimators_understate_variance_when_the_market_gaps():
    bars, _ = data.synthetic_ohlc(n_years=12, overnight_share=0.4, seed=965)
    tbl = st.bias_table(bars)
    for c in st.GAP_BLIND:
        assert tbl.loc[c, "ratio_to_cc"] < 0.85
    # Yang-Zhang is built to contain the gap, so it must land near the close-to-close level.
    assert 0.85 < tbl.loc["yang_zhang", "ratio_to_cc"] < 1.15


def test_overnight_share_recovers_the_planted_split():
    bars, _ = data.synthetic_ohlc(n_years=15, overnight_share=0.35, seed=965)
    assert st.overnight_share(bars) == pytest.approx(0.35, abs=0.06)


def test_discretisation_biases_the_range_low_and_more_sampling_fixes_it():
    """A finitely-sampled path has a smaller range than the continuous one it approximates.

    This is Marsh & Rosenfeld's (1986) non-trading bias, reproduced: on a gapless world the
    range estimators read *below* the truth, and the gap closes as the path is sampled more
    finely. Real tapes have the same problem for the same reason, which is one more reason a
    range estimator is not the free lunch the theorem promises.
    """
    ratios = {}
    for n_steps in (26, 78, 400):
        bars, tr = data.synthetic_ohlc(n_years=10, overnight_share=0.0,
                                       n_intraday=n_steps, seed=965)
        tbl = st.efficiency_table(bars, tr["sigma"])
        ratios[n_steps] = tbl.loc["parkinson", "mean_ratio_to_truth"]
    assert ratios[26] < ratios[78] < ratios[400] < 1.05
    assert ratios[400] > 0.9
    # Close-to-close, which uses no range at all, is unbiased whatever the sampling.
    bars, tr = data.synthetic_ohlc(n_years=10, overnight_share=0.0, n_intraday=26, seed=965)
    cc = st.efficiency_table(bars, tr["sigma"]).loc["close_close", "mean_ratio_to_truth"]
    assert cc == pytest.approx(1.0, abs=0.25)


# --------------------------------------------------------------------------- #
# Rolling forms and the forecast race
# --------------------------------------------------------------------------- #
def test_rolling_variance_averages_the_single_day_estimators(planted):
    bars, _ = planted
    roll = st.rolling_variance(bars, window=21)
    raw = st.all_estimators(bars, 21)
    assert np.allclose(roll["parkinson"].dropna(),
                       raw["parkinson"].rolling(21).mean().dropna())
    assert np.allclose(roll["yang_zhang"].dropna(), raw["yang_zhang"].dropna())


def test_annualised_vol_is_in_believable_units(planted):
    bars, tr = planted
    v = st.annualised_vol(st.rolling_variance(bars)["close_close"]).dropna()
    assert 0.05 < v.median() < 0.5
    assert v.median() == pytest.approx(tr["vol_ann"], abs=0.06)


def test_qlike_is_zero_only_for_a_perfect_forecast():
    a = pd.Series([1e-4, 4e-4, 9e-4])
    assert st.qlike(a, a).abs().max() == pytest.approx(0.0, abs=1e-12)
    assert (st.qlike(a, a * 2) > 0).all()
    assert (st.qlike(a, a * 0.5) > 0).all()


def test_diebold_mariano_signs_and_null():
    rng = np.random.default_rng(965)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    a = pd.Series(rng.normal(1.0, 0.2, 2000), index=idx)
    b = pd.Series(rng.normal(1.0, 0.2, 2000), index=idx)
    assert abs(st.diebold_mariano(a, b)["dm"]) < 2.5          # two equal models
    worse = a + 0.2
    assert st.diebold_mariano(worse, a)["dm"] > 2.0           # A is worse -> positive


def test_diebold_mariano_hac_beats_the_naive_version_on_overlapping_losses():
    """Overlapping windows make losses autocorrelated; the HAC lag must widen the SE."""
    rng = np.random.default_rng(965)
    base = pd.Series(rng.normal(0.02, 0.1, 3000),
                     index=pd.bdate_range("2010-01-01", periods=3000)).rolling(21).mean()
    zero = pd.Series(0.0, index=base.index)
    hac = abs(st.diebold_mariano(base, zero)["dm"])
    naive = abs(st.diebold_mariano(base, zero, lags=0)["dm"])
    assert hac < naive


def test_forecast_race_runs_and_ranks(planted):
    bars, _ = planted
    tbl = st.forecast_race(bars, burn=252)
    assert set(tbl.index) == set(st.ESTIMATORS)
    assert (tbl["qlike"] > 0).all()
    assert tbl.loc["close_close", "dm_vs_cc"] == pytest.approx(0.0, abs=1e-9) or np.isnan(
        tbl.loc["close_close", "dm_vs_cc"])


def test_rescaling_removes_the_level_penalty():
    """A gap-blind estimator must score better once its level error is corrected."""
    bars, _ = data.synthetic_ohlc(n_years=15, overnight_share=0.4, seed=965)
    raw = st.forecast_race(bars, burn=252)
    scaled = st.scaled_forecast_race(bars, burn=252)
    assert scaled.loc["parkinson", "qlike"] < raw.loc["parkinson", "qlike"]
    assert scaled.loc["parkinson", "scale"] > 1.0


def test_scale_factors_come_only_from_the_burn_in(planted):
    """No look-ahead: the same scale must come out whatever happens after the burn-in."""
    bars, _ = planted
    cut = bars.copy()
    cut.iloc[400:, cut.columns.get_loc("high")] *= 1.5   # mangle the future
    a = st.scaled_forecast_race(bars, burn=300)["scale"]
    b = st.scaled_forecast_race(cut, burn=300)["scale"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"efficiency_parkinson": 4.5, "efficiency_gk": 5.0, "efficiency_rs": 3.5,
         "overnight_share_spy": 0.4, "ratio_parkinson_spy": 0.6, "tickers": list("ABCDE"),
         "n_qlike_wins": 4, "pooled_dm": 3.0, "best_estimator": "garman_klass",
         "best_qlike_gain": 0.05}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(efficiency_parkinson=2.0))["signal"] == "Weak"
    assert st.verdict(_headline(efficiency_parkinson=1.1))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(pooled_dm=0.5))["trad"] == "Fragile"
    assert st.verdict(_headline(n_qlike_wins=1))["trad"] == "Mirage"


def test_verdict_quotes_its_inputs():
    v = st.verdict(_headline(overnight_share_spy=0.37))
    assert "37%" in v["signal_why"] and "37%" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
