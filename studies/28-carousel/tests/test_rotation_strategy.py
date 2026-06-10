"""The rotation engine recovers the baked sector momentum (and finds none on the null), and the rotation
book beats the equal-weight basket with momentum, not without it."""

import numpy as np

from carousel import data, rotation, strategy, decompose, extension


def test_deterministic_and_momentum(mom):
    panel, truth = mom
    panel2, _ = data.synthetic_sectors(mom_strength=0.0011, seed=28)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())
    assert truth.has_momentum


def test_rotation_strength_recovers(mom_panel):
    rs = rotation.rotation_strength(mom_panel)
    assert rs["top_minus_bottom_ann_pct"] > 5.0
    assert rs["momentum_present"]


def test_rotation_strength_flat_on_null(null_panel):
    rs = rotation.rotation_strength(null_panel)
    assert abs(rs["top_minus_bottom_ann_pct"]) < 8.0


def test_rotation_beats_basket_with_momentum(mom_panel):
    cmp = strategy.compare(mom_panel, cost_bps=3.0)
    assert cmp["rotation_minus_ew_sharpe"] > 0.2
    a = decompose.vs_equal_weight(mom_panel, cost_bps=3.0)
    assert a["alpha_t"] > 3.0


def test_rotation_no_edge_on_null(null_panel):
    cmp = strategy.compare(null_panel, cost_bps=3.0)
    assert cmp["rotation_minus_ew_sharpe"] < 0.2


def test_topk_sweep_generalises_on_momentum_not_null(mom_panel, null_panel):
    sw_m = extension.topk_sweep(mom_panel, cost_bps=3.0)
    sw_n = extension.topk_sweep(null_panel, cost_bps=3.0)
    assert sw_m["frac_beat_ew"] > 0.5            # real momentum helps across most k
    assert sw_m["mean_gain"] > sw_n["mean_gain"]  # and out-earns the null on average
