"""The decomposition recovers the baked-in truth: the σ-transform moves zero signal within a
length (adaptive == a constant), the cheat-sheet is pure arithmetic in n, σ-calibration never
beats a re-optimised constant, and on a single-horizon tape Rescaled long-RSI adds ~nothing."""

import math

import numpy as np

from sigma_sleight import decompose


# 1 · the transform is a monotone relabel ------------------------------------

def test_mapping_is_strictly_monotone():
    for n in (2, 14, 200):
        chk = decompose.monotone_check(n)
        assert chk["strictly_increasing"]
        assert chk["min_step"] > 0
        assert chk["max_roundtrip_err"] < 1e-6


def test_adaptive_band_is_identical_to_a_constant(close):
    """The load-bearing result: σ-zone trades == constant-RSI-band trades, to the bar."""
    for n, ls in ((2, -1.732), (14, -1.0)):
        out = decompose.crossing_identity(close, n, ls)
        assert out["max_position_diff"] == 0.0
        assert out["identical"]
        assert out["implied_lower_rsi"] < 50.0


# 2 · the cheat-sheet is arithmetic ------------------------------------------

def test_zone_table_is_arithmetic_and_compresses():
    z = decompose.zone_arithmetic([2, 14, 50, 200])
    assert z["rsi70_is_1p53_sigma"]
    assert math.isclose(z["sigma_at_rsi70_len14"], 1.527, abs_tol=0.01)
    assert z["compresses_toward_50"]


# 3 · σ-calibration never beats a re-optimised constant ----------------------

def test_adaptive_does_not_beat_reoptimised_constant(close):
    """σ-calibration is just one constant the grid also searches, so it can't win — its edge over
    the in-sample-best constant is <= 0 (up to the grid's own resolution)."""
    cmp = decompose.strategy_compare(close, length=2, cost_bps=1.0)
    assert cmp["adaptive_vs_reopt_sharpe"] <= 1e-9


def test_real_edge_exists_for_some_band(close):
    """Sanity: the baked-in mean reversion means *some* oversold band is profitable gross of the
    multiple-testing — otherwise we'd be testing noise."""
    cmp = decompose.strategy_compare(close, length=2, cost_bps=0.0)
    assert cmp["reopt"]["sharpe"] > 0.2


# 4 · rescaling is a cross-length monotone relabel ---------------------------

def test_rescaling_is_rank_invariant(close):
    """The baked-in identity: rescaled RSI(long) shares the rank IC of *raw* RSI(long), because
    rescaling is a monotone relabel — it cannot add or remove rank/threshold information."""
    out = decompose.rescale_increment(close, target_length=14, long_length=70, horizon=5)
    assert abs(out["rescale_ic_gap"]) < 1e-9          # identical to the raw long RSI's IC
    assert out["n"] > 100


def test_window_increment_is_real_but_not_from_sigma(close):
    """The longer window genuinely adds information over native short RSI — but that increment is
    present in raw RSI(long) already; the σ-rescaling contributes none of it (previous test)."""
    out = decompose.rescale_increment(close, target_length=14, long_length=70, horizon=5)
    assert abs(out["incremental_ic_window_over_native"]) > 0.03


# 5 · the Reality-Check panels — the pre-registered bar's plumbing ------------

def test_race_panel_shapes_and_margin_identity(close):
    """The RC universes are built from the single payoff definition: the declared panel carries
    adaptive+fixed per length, the full panel adds every reopt-grid constant, and the margin
    panel is exactly adaptive − fixed, column for column."""
    lengths = (2, 5, 14)
    declared = decompose.race_panel(close, lengths=lengths, cost_bps=1.0)
    assert list(declared.columns) == [c for n in lengths for c in (f"adaptive_n{n}", f"fixed_n{n}")]
    full = decompose.race_panel(close, lengths=lengths, cost_bps=1.0, include_grid=True)
    assert full.shape[1] == len(lengths) * (2 + 40)   # named variants + lower 5..44 grid
    margins = decompose.margin_panel(close, lengths=lengths, cost_bps=1.0)
    for n in lengths:
        np.testing.assert_allclose(
            margins[f"margin_n{n}"], declared[f"adaptive_n{n}"] - declared[f"fixed_n{n}"])


def test_margin_reality_check_is_dead_even_with_a_real_edge(close):
    """The study's mirage line, on the synthetic positive control: the tape HAS a real oversold
    edge (test_real_edge_exists_for_some_band), yet the σ-band's margin over fixed 30/50 still
    fails the White RC — the σ-calibration adds nothing even when there is something to add."""
    from quantlab import bayes
    margins = decompose.margin_panel(close, lengths=(2, 5, 14), cost_bps=1.0)
    rc = bayes.reality_check(margins, n_boot=200, seed=0)
    assert rc["n_series"] == 3
    assert rc["reality_check_pvalue"] > 0.10
