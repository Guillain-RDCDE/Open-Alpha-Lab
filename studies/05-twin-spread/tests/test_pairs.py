"""Pair formation: normalization and the minimum-SSD selector."""

import numpy as np

from pairs_trading import pairs, robustness


def test_normalized_prices_start_at_one(panel):
    norm = pairs.normalized_prices(panel.iloc[:252])
    assert np.allclose(norm.iloc[0].to_numpy(), 1.0)


def test_selector_recovers_true_twins(panel, true_pairs):
    """The top-N minimum-SSD pairs should be exactly the baked-in twins."""
    selected = pairs.select_pairs(panel.iloc[:252], top_n=len(true_pairs))
    recall = robustness.selection_recall(selected, true_pairs)
    assert recall >= 0.8        # finds at least 5 of 6 true twins on the first year


def test_pairs_sorted_by_ssd(panel):
    selected = pairs.select_pairs(panel.iloc[:252], top_n=10)
    ssds = [p.ssd for p in selected]
    assert ssds == sorted(ssds)
    assert all(p.sigma > 0 for p in selected)


def test_partial_history_names_are_ineligible(panel):
    """A name with NaNs across the formation window can't be paired."""
    p = panel.copy()
    p.iloc[:50, p.columns.get_loc("NS00")] = np.nan
    selected = pairs.select_pairs(p.iloc[:252], top_n=50)
    assert all("NS00" not in (pp.a, pp.b) for pp in selected)
