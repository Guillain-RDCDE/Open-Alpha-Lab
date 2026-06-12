"""The gauntlet must recover known structure offline and produce well-formed verdicts."""

import numpy as np

from true_strength import collinearity as col


def test_oscillators_agree_more_on_structure_than_noise(synthetic):
    """Machinery control, part 1: positions agree a little more where structure is planted."""
    frames, truth = synthetic
    rec = col.structure_recall(frames, truth)
    assert rec["agree_structured"] > rec["agree_noise"]
    assert rec["agree_structured"] > 0.5


def test_level_collinearity_is_mechanical(synthetic):
    """Machinery control, part 2 (the load-bearing one): the TSI~MACD level collinearity
    and the spanning R² are about as high on pure random walks as on planted structure —
    the co-movement is a property of the filters, not of the input."""
    frames, truth = synthetic
    rec = col.structure_recall(frames, truth)
    assert rec["corr_tsi_macd_noise"] > 0.7          # high on NOISE — mechanical
    assert rec["spanning_r2_noise"] > 0.5            # spanned even with nothing planted
    # noise is within shouting distance of structure (not an order of magnitude below)
    assert rec["corr_tsi_macd_noise"] > rec["corr_tsi_macd_structured"] - 0.15


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
    assert rc["benchmark"] == "zero"


def test_reality_check_excess_of_buy_and_hold(synthetic):
    """The beta-stripped Reality Check scores each variant against EW buy-and-hold."""
    frames, _ = synthetic
    rc = col.reality_check_grid(frames, n_boot=200, excess_of_bh=True)
    assert rc["benchmark"] == "buy_and_hold"
    assert 0.0 <= rc["reality_check_pvalue"] <= 1.0
    assert np.isfinite(rc["bh_sharpe"])


def test_cost_sweep_monotone_in_cost(synthetic):
    frames, _ = synthetic
    sweep = col.cost_sweep(frames)
    assert sweep["mean_net_bps"].is_monotonic_decreasing


def test_orthogonalised_residual_well_formed(synthetic):
    frames, _ = synthetic
    r = col.orthogonalised_tsi_edge(frames, cost_bps=10.0)
    assert r["n_names"] > 0
    # Gross is the pre-declared information criterion; net is the friction statement.
    assert np.isfinite(r["residual_gross_sharpe"])
    assert np.isfinite(r["residual_gross_hac_t"])
    assert r["residual_gross_sharpe_se"] > 0.0
    assert np.isfinite(r["residual_net_sharpe"])
    assert np.isfinite(r["raw_tsi_ls_gross_sharpe"])
    assert np.isfinite(r["raw_tsi_ls_net_sharpe"])
    # Costs can only hurt, and the residual book churns more than the raw level book.
    assert r["residual_net_sharpe"] <= r["residual_gross_sharpe"]
    assert r["residual_ann_turnover"] > r["raw_ann_turnover"]
