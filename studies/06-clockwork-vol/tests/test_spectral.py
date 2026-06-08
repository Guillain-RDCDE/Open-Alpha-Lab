"""The detector must light up on a real cycle and stay dark on red noise."""

import numpy as np

from vix_cycles import robustness, spectral


def test_periodogram_peaks_at_injected_period(synth):
    series, injected = synth
    periods, power = spectral.periodogram(series)
    # The tallest peak in the cycle band should sit near one of the injected periods.
    band = (20.0, 200.0)
    m = (periods >= band[0]) & (periods <= band[1])
    top = periods[m][np.argmax(power[m])]
    assert min(abs(top - c.period) for c in injected) <= 0.2 * 80.0


def test_injected_cycles_clear_red_noise_envelope(synth):
    series, injected = synth
    env = spectral.red_noise_envelope(series, n_sim=400, seed=0)
    recall = robustness.detection_recall(env, injected, q=0.99)
    assert recall >= 0.5        # at least the dominant injected cycle is significant at 1%


def test_pure_red_noise_has_no_significant_claimed_cycle(red_noise):
    """On structureless red noise, the claimed 80-day cycle must NOT be significant."""
    r = spectral.test_period(red_noise, 80.0, n_sim=400, seed=1)
    assert r["p_value"] > 0.05


def test_injected_period_is_significant(synth):
    series, _ = synth
    r = spectral.test_period(series, 80.0, n_sim=400, seed=0)
    assert r["p_value"] < 0.05


def test_ar1_rho_recovers_persistence(red_noise):
    # The surrogate was built with rho=0.94; the estimator should land close.
    assert 0.85 < spectral.ar1_rho(red_noise) < 0.99
