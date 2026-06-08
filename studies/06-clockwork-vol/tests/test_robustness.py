"""Period stability, the red-noise p-value table, and the offline detection check."""

import numpy as np

from vix_cycles import robustness, spectral


def test_period_stability_tight_on_fixed_cycle(synth):
    series, injected = synth
    stab = robustness.period_stability(series, band=(20.0, 200.0), window=1000, step=200)
    # A genuinely fixed cycle ⇒ the rolling dominant period barely moves.
    assert stab["dominant_period"].std() < 0.25 * 80.0


def test_red_noise_pvalues_table_shape(synth):
    series, _ = synth
    tbl = robustness.red_noise_pvalues(series, targets=(40.0, 80.0), n_sim=200, seed=0)
    assert list(tbl.index) == [40.0, 80.0]
    assert (tbl["p_value"] <= 1.0).all() and (tbl["p_value"] >= 0.0).all()
    # The injected 80d cycle should be significant here.
    assert tbl.loc[80.0, "p_value"] < 0.05


def test_detection_recall_one_on_synthetic(synth):
    series, injected = synth
    env = spectral.red_noise_envelope(series, n_sim=400, seed=0)
    assert robustness.detection_recall(env, injected, q=0.95) >= 0.5
