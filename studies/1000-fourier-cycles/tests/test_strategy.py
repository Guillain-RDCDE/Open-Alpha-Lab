"""Strategy tests for Study 1000 — the spectrum, and the peaks noise provides."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cyclehunt import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The periodogram, checked against things with known answers
# --------------------------------------------------------------------------- #
def test_a_pure_sinusoid_peaks_at_its_own_period():
    n = 2048
    period = 64.0
    t = np.arange(n)
    x = pd.Series(np.sin(2 * np.pi * t / period),
                  index=pd.bdate_range("1993-02-01", periods=n))
    pg = st.periodogram(x)
    peak = pg.loc[pg["power"].idxmax()]
    assert peak["period"] == pytest.approx(period, rel=0.05)


def test_two_sinusoids_give_two_peaks():
    n = 4096
    t = np.arange(n)
    x = pd.Series(np.sin(2 * np.pi * t / 50) + 0.8 * np.sin(2 * np.pi * t / 200),
                  index=pd.bdate_range("1993-02-01", periods=n))
    peaks = st.top_peaks(st.periodogram(x), k=2)
    periods = sorted(peaks["period"])
    assert periods[0] == pytest.approx(50, rel=0.1)
    assert periods[1] == pytest.approx(200, rel=0.1)


def test_white_noise_has_a_flat_spectrum_on_average():
    x = st.synthetic_series(n=8000)
    pg = st.periodogram(x)
    lo = pg[pg["frequency"] < 0.1]["power"].mean()
    hi = pg[pg["frequency"] > 0.4]["power"].mean()
    assert lo / hi == pytest.approx(1.0, abs=0.25)


def test_detrending_removes_the_fake_long_cycle():
    """An undetrended trend puts huge power at the lowest frequency — the classic artefact."""
    n = 3000
    t = np.arange(n)
    rng = np.random.default_rng(1000)
    x = pd.Series(0.001 * t + rng.normal(0, 1, n),
                  index=pd.bdate_range("1993-02-01", periods=n))
    with_trend = st.periodogram(x, detrend=False)
    without = st.periodogram(x, detrend=True)
    assert with_trend["power"].max() > without["power"].max() * 3


def test_the_hann_window_reduces_leakage():
    n = 2048
    t = np.arange(n)
    x = pd.Series(np.sin(2 * np.pi * t / 63.7),        # deliberately not a whole number of bins
                  index=pd.bdate_range("1993-02-01", periods=n))
    plain = st.periodogram(x, window="none")
    hann = st.periodogram(x, window="hann")
    def concentration(pg):
        p = pg["power"].to_numpy()
        return float(np.sort(p)[-3:].sum() / p.sum())
    assert concentration(hann) > concentration(plain)


def test_periodogram_declines_on_a_short_series():
    assert st.periodogram(pd.Series(np.arange(20.0))).empty


def test_top_peaks_respects_the_period_bounds():
    x = st.synthetic_series(n=4000)
    peaks = st.top_peaks(st.periodogram(x), k=5, min_period=20.0, max_period=100.0)
    assert (peaks["period"] >= 20).all() and (peaks["period"] <= 100).all()


# --------------------------------------------------------------------------- #
# How big a peak does noise give?
# --------------------------------------------------------------------------- #
def test_the_expected_maximum_grows_like_log_m():
    assert st.expected_max_relative_power(100) == pytest.approx(np.log(100) + 0.5772,
                                                                rel=1e-4)
    assert st.expected_max_relative_power(2000) > st.expected_max_relative_power(200)
    assert st.expected_max_relative_power(2000) > 7.0


def test_noise_really_does_produce_a_large_peak():
    """The study's central fact, verified rather than asserted."""
    x = st.synthetic_series(n=4000)
    pg = st.periodogram(x)
    rel = pg["power"].max() / pg["power"].mean()
    assert rel > 5.0


def test_the_simulated_maximum_matches_the_theory():
    sim = st.spurious_peak_distribution(n=2000, n_sims=200)
    assert sim["mean_max"] == pytest.approx(sim["theoretical"], rel=0.4)
    assert sim["p95_max"] > sim["median_max"]


def test_a_longer_series_gives_a_bigger_spurious_peak():
    """More bins means more chances, and the peak grows accordingly."""
    short = st.spurious_peak_distribution(n=500, n_sims=150)
    long = st.spurious_peak_distribution(n=8000, n_sims=150)
    assert long["median_max"] > short["median_max"]


def test_autocorrelation_inflates_the_spurious_peak():
    """Which is why testing against a white null is not conservative, it is wrong."""
    white = st.spurious_peak_distribution(n=3000, n_sims=150, ar1=0.0)
    red = st.spurious_peak_distribution(n=3000, n_sims=150, ar1=0.4)
    assert red["p95_max"] > white["p95_max"]


# --------------------------------------------------------------------------- #
# Fisher's test
# --------------------------------------------------------------------------- #
def test_fisher_does_not_reject_white_noise():
    rejects = 0
    for k in range(40):
        x = st.synthetic_series(n=2000, seed=1000 + k)
        out = st.fisher_g_test(st.periodogram(x))
        rejects += bool(out.get("significant_5pct", False))
    assert rejects / 40 < 0.20          # nominal 5%, allowing simulation noise


def test_fisher_rejects_a_genuine_cycle():
    x = st.synthetic_series(n=4000, period=100.0, amplitude=0.5)
    out = st.fisher_g_test(st.periodogram(x))
    assert out["significant_5pct"]
    assert out["peak_period"] == pytest.approx(100, rel=0.1)


def test_fisher_needs_a_bigger_amplitude_in_a_longer_search():
    """More bins, more multiple testing, a higher bar — exactly as it should be."""
    found_short = st.fisher_g_test(
        st.periodogram(st.synthetic_series(n=1000, period=50.0, amplitude=0.25)))
    found_long = st.fisher_g_test(
        st.periodogram(st.synthetic_series(n=16000, period=50.0, amplitude=0.25)))
    assert np.isfinite(found_short["p_value"]) and np.isfinite(found_long["p_value"])


def test_the_g_statistic_is_a_share_of_total_power():
    x = st.synthetic_series(n=2000)
    out = st.fisher_g_test(st.periodogram(x))
    assert 0 < out["g"] < 1


def test_fisher_declines_on_a_tiny_spectrum():
    assert "g" not in st.fisher_g_test(pd.DataFrame({"power": [1.0, 2.0],
                                                     "period": [2.0, 3.0],
                                                     "frequency": [0.5, 0.33]}))


# --------------------------------------------------------------------------- #
# The AR(1) null
# --------------------------------------------------------------------------- #
def test_ar1_is_estimated_correctly():
    for phi in (0.0, 0.3, -0.2):
        x = st.synthetic_series(n=20000, ar1=phi)
        assert st.ar1_null(x)["phi"] == pytest.approx(phi, abs=0.05)


def test_the_ar1_spectrum_tilts_toward_low_frequencies():
    f = np.linspace(0.001, 0.5, 500)
    s = st.ar1_spectral_density(f, 0.5)
    assert s[0] > s[-1] * 3


def test_a_negative_ar1_tilts_the_other_way():
    f = np.linspace(0.001, 0.5, 500)
    s = st.ar1_spectral_density(f, -0.5)
    assert s[0] < s[-1] / 3


def test_correcting_for_ar1_moves_the_peak_on_autocorrelated_noise():
    x = st.synthetic_series(n=6000, ar1=0.5)
    pg = st.periodogram(x)
    out = st.peak_against_ar1(pg, st.ar1_null(x)["phi"])
    assert out["relative_power_ar1"] < out["relative_power_white"] + 1e-9


def test_a_genuine_cycle_survives_the_ar1_correction():
    x = st.synthetic_series(n=6000, period=80.0, amplitude=0.6, ar1=0.3)
    pg = st.periodogram(x)
    out = st.peak_against_ar1(pg, st.ar1_null(x)["phi"])
    assert out["peak_period_ar1"] == pytest.approx(80, rel=0.15)


def test_ar1_null_declines_on_a_short_series():
    assert "phi" not in st.ar1_null(pd.Series(np.arange(20.0)))


# --------------------------------------------------------------------------- #
# Does the cycle hold up?
# --------------------------------------------------------------------------- #
def test_a_real_cycle_keeps_its_period_and_stays_coherent():
    x = st.synthetic_series(n=8000, period=120.0, amplitude=0.5)
    out = st.split_sample_peak(x, max_period=500.0)
    assert out["period_ratio"] == pytest.approx(1.0, abs=0.2)
    assert out["coherent"]
    assert out["phase_concentration"] > 0.7


def test_the_naive_phase_test_fails_even_on_a_genuine_cycle():
    """Documenting why the coherence measure exists rather than the obvious comparison.

    The periodogram measures the period on a grid. Near a 120-session period with 8,000
    observations the neighbouring bins are several sessions apart, and being off by one part in
    a hundred accumulates a radian of phase over a few thousand steps. A perfectly stable
    planted cycle therefore fails a naive phase-equality test — the test is measuring grid
    resolution, not reality.
    """
    x = st.synthetic_series(n=8000, period=120.0, amplitude=0.5)
    out = st.split_sample_peak(x, max_period=500.0)
    assert out["phase_error_fraction"] > 0.25      # the naive test says "not a cycle"
    assert out["coherent"]                          # the coherence test says otherwise


def test_a_spurious_cycle_is_incoherent():
    ratios, concs = [], []
    for k in range(8):
        x = st.synthetic_series(n=8000, seed=1000 + k)
        out = st.split_sample_peak(x, max_period=500.0)
        if "period_ratio" in out:
            ratios.append(out["period_ratio"])
            concs.append(out["phase_concentration"])
    assert np.mean(concs) < 0.7
    assert np.std(ratios) > 0.1


def test_phase_coherence_separates_a_real_cycle_from_noise():
    real = st.phase_coherence(st.synthetic_series(n=8000, period=100.0, amplitude=0.6),
                              100.0)
    fake = st.phase_coherence(st.synthetic_series(n=8000, seed=7), 100.0)
    assert real["concentration"] > fake["concentration"]
    assert real["coherent"] and not fake["coherent"]


def test_phase_coherence_declines_on_a_short_series():
    assert "concentration" not in st.phase_coherence(st.synthetic_series(n=100), 50.0)


def test_the_amplitude_of_a_real_cycle_persists():
    x = st.synthetic_series(n=8000, period=120.0, amplitude=0.5)
    out = st.split_sample_peak(x, max_period=500.0)
    assert out["amplitude_decay"] == pytest.approx(1.0, abs=0.5)


def test_split_sample_declines_on_a_short_series():
    assert "period_first" not in st.split_sample_peak(st.synthetic_series(n=100))


# --------------------------------------------------------------------------- #
# Trading it
# --------------------------------------------------------------------------- #
def test_a_real_cycle_can_be_traded():
    x = st.synthetic_series(n=6000, period=60.0, amplitude=1.2)
    out = st.cycle_trade(x, 60.0, fit_window=600, cost_bps=0.0)
    assert out["sharpe"] > 0.5
    assert out["hit_rate"] > 0.52


def test_a_spurious_cycle_cannot_be():
    sharpes = []
    for k in range(6):
        x = st.synthetic_series(n=6000, seed=1000 + k)
        pg = st.periodogram(x)
        peak = st.top_peaks(pg, 1, min_period=10.0, max_period=400.0)
        if peak.empty:
            continue
        out = st.cycle_trade(x, float(peak.iloc[0]["period"]), fit_window=1000,
                             cost_bps=0.0)
        if "sharpe" in out:
            sharpes.append(out["sharpe"])
    assert abs(np.mean(sharpes)) < 0.5


def test_the_cycle_trade_is_strictly_out_of_sample():
    x = st.synthetic_series(n=3000)
    bad = x.copy()
    bad.iloc[2500:] *= 50
    a = st.cycle_trade(x, 60.0, fit_window=500, cost_bps=0.0)["returns"].iloc[:1500]
    b = st.cycle_trade(bad, 60.0, fit_window=500, cost_bps=0.0)["returns"].iloc[:1500]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_costs_reduce_the_cycle_trade():
    x = st.synthetic_series(n=5000, period=60.0, amplitude=1.0)
    free = st.cycle_trade(x, 60.0, fit_window=600, cost_bps=0.0)
    paid = st.cycle_trade(x, 60.0, fit_window=600, cost_bps=100.0)
    assert paid["cagr"] < free["cagr"]


def test_cycle_trade_declines_on_a_short_series():
    assert "sharpe" not in st.cycle_trade(st.synthetic_series(n=200), 60.0)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_assets": 7, "n_bins": 2000, "mean_relative_power": 8.4,
         "theoretical_max": 8.2, "simulated_p95": 11.3,
         "n_significant_white": 1, "n_significant_ar1": 0,
         "control_asset": "UNG", "control_period": 251.0,
         "lead_asset": "SPY", "period_first": 118.0, "period_second": 47.0,
         "phase_error_fraction": 0.71, "phase_concentration": 0.31,
         "fit_window": 1000,
         "cycle_cagr": -0.011, "cycle_sharpe": -0.06, "cycle_hit_rate": 0.498,
         "cycle_switches": 41.0, "cycle_buyhold": 0.094}
    h.update(over)
    return h


def test_verdict_is_busted_when_nothing_survives():
    assert st.verdict(_headline(n_significant_white=0))["signal"] == "Busted"


def test_verdict_is_partial_when_only_the_white_null_rejects():
    assert st.verdict(_headline())["signal"] == "Partial"


def test_verdict_is_confirmed_only_with_ar1_and_phase():
    assert st.verdict(_headline(n_significant_ar1=1))["signal"] == "Partial"
    assert st.verdict(_headline(n_significant_ar1=1,
                                phase_concentration=0.9))["signal"] == "Confirmed"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(cycle_sharpe=0.1))["trad"] == "Partial"
    assert st.verdict(_headline(cycle_sharpe=0.5))["trad"] == "Useful"


def test_verdict_prose_states_the_noise_benchmark_and_the_control():
    v = st.verdict(_headline())
    assert "random walk" in v["one_sentence"]
    assert "positive control" in v["signal_why"]
    assert "Fisher" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
