"""The gauntlet must recover known structure offline and produce well-formed verdicts."""

import numpy as np

from true_strength import collinearity as col


def test_oscillators_agree_more_on_structure_than_noise(synthetic):
    """Machinery sanity: on trend/cycle names the three positions agree far more often."""
    frames, truth = synthetic
    rec = col.structure_recall(frames, truth)
    assert rec["agree_structured"] > rec["agree_noise"]
    assert rec["agree_structured"] > 0.5


def test_level_collinearity_is_high_on_momentum(synthetic):
    """On a momentum market the z-scored oscillators are strongly positively correlated."""
    frames, truth = synthetic
    structured = {t.ticker for t in truth}
    lc = col.level_collinearity({k: v for k, v in frames.items() if k in structured})
    assert (lc["median_corr"] > 0.5).all()


def test_incremental_information_r2_bounded(synthetic):
    frames, _ = synthetic
    inc = col.incremental_information(frames)
    assert 0.0 <= inc["pooled_r2"] <= 1.0
    assert 0.0 <= inc["median_name_r2"] <= 1.0


def test_reality_check_pvalue_is_a_probability(synthetic):
    frames, _ = synthetic
    rc = col.reality_check_grid(frames, n_boot=200)
    assert 0.0 <= rc["reality_check_pvalue"] <= 1.0
    assert rc["n_variants"] == 24


def test_cost_sweep_monotone_in_cost(synthetic):
    frames, _ = synthetic
    sweep = col.cost_sweep(frames)
    assert sweep["mean_net_bps"].is_monotonic_decreasing


def test_orthogonalised_residual_well_formed(synthetic):
    frames, _ = synthetic
    r = col.orthogonalised_tsi_edge(frames, cost_bps=10.0)
    assert r["n_names"] > 0
    # Sharpes are finite numbers (the residual edge is the object the verdict rests on).
    assert np.isfinite(r["residual_sharpe"])
    assert np.isfinite(r["raw_tsi_ls_sharpe"])
