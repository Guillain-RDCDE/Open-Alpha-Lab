"""The theorist's machinery: bandpass, phase, turning points, fixed-period projection."""

import numpy as np

from vix_cycles import cycles


def test_bandpass_recovers_injected_amplitude(synth):
    series, injected = synth
    dominant = max(injected, key=lambda c: c.amplitude)
    band = cycles.bandpass(series, dominant.period, width_frac=0.4)
    # The extracted component should swing on the order of the injected amplitude.
    assert 0.5 * dominant.amplitude < band.std() < 3.0 * dominant.amplitude


def test_turning_points_spaced_by_about_a_period(synth):
    series, injected = synth
    dominant = max(injected, key=lambda c: c.amplitude)
    tp = cycles.turning_points(series, dominant.period)
    lows = tp["lows"]
    assert len(lows) >= 3
    gaps = np.diff(lows.values).astype("timedelta64[D]").astype(int)
    # Consecutive lows roughly one period apart (calendar days ≳ business-day period).
    assert np.median(gaps) > dominant.period * 0.6


def test_fit_and_project_are_self_consistent(synth):
    series, injected = synth
    P = injected[0].period
    fit = cycles.fit_sinusoid(series.iloc[:1000], P)
    # Projecting 0 steps reproduces the model value at the last fitted sample.
    val0 = cycles.project(fit, 0)[0]
    w = 2 * np.pi / P
    expected = fit["c0"] + fit["A"] * np.cos(w * (fit["n"] - 1)) + fit["B"] * np.sin(w * (fit["n"] - 1))
    assert abs(val0 - expected) < 1e-9


def test_next_turn_returns_a_future_step(synth):
    series, injected = synth
    fit = cycles.fit_sinusoid(series.iloc[:1000], injected[0].period)
    k = cycles.next_turn(fit, kind="low")
    assert 1 <= k <= int(injected[0].period) + 2
