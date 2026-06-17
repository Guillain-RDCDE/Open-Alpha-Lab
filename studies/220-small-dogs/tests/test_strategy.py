"""Strategy tests for Study 220 (Small Dogs of the Dow) — offline/synthetic only."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from small_dogs import data, strategy  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests — hac_tstat and block_bootstrap_ci
# ---------------------------------------------------------------------------
def test_hac_tstat_strong_signal():
    """A near-constant positive series should yield a very high t-stat."""
    x = np.full(40, 0.05) + np.random.default_rng(0).normal(0, 0.001, 40)
    assert strategy.hac_tstat(x) > 5


def test_hac_tstat_zero_mean():
    """A zero-mean series should yield a t-stat near zero."""
    rng = np.random.default_rng(99)
    x = rng.normal(0, 0.05, 100)
    assert abs(strategy.hac_tstat(x)) < 3


def test_block_bootstrap_ci_brackets_zero():
    """For a noise series the 95% CI should straddle zero and p > 0.10."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, 30)
    lo, hi, p = strategy.block_bootstrap_ci(x, n_boot=2000)
    assert lo < 0 < hi
    assert p > 0.10


def test_summarize_basic():
    """summarize() should correctly compound a constant-return series."""
    ann = pd.DataFrame({
        "small_dogs": [0.10, 0.10],
        "dow": [0.0, 0.0],
    }, index=[2001, 2002])
    s = strategy.summarize(ann, col="small_dogs", bench="dow")
    assert abs(s["strat_cagr"] - 0.10) < 1e-9
    assert s["win_rate"] == 1.0
    assert abs(s["total_strat"] - 1.21) < 1e-9


def test_alpha_beta_recovers_planted_line():
    """OLS should recover alpha=0.02 beta=1.5 from a deterministic panel."""
    rng = np.random.default_rng(2)
    x = rng.normal(0.08, 0.15, 200)
    y = 0.02 + 1.5 * x + rng.normal(0, 1e-6, 200)
    ann = pd.DataFrame({"small_dogs": y, "dow": x})
    ab = strategy.alpha_beta(ann, strat_col="small_dogs", bench_col="dow")
    assert abs(ab["alpha"] - 0.02) < 1e-3
    assert abs(ab["beta"] - 1.5) < 1e-3


# ---------------------------------------------------------------------------
# Spine tests — planted / null controls on the synthetic panel
# ---------------------------------------------------------------------------
def test_spine_planted_small_dogs_beats_bench(planted_panel):
    """With a planted edge, the Small Dogs MUST beat the equal-weight index."""
    ann = strategy.annual_returns_from_synthetic(planted_panel)
    t = strategy.hac_tstat(ann["excess_small"].to_numpy())
    s = strategy.summarize(ann, col="small_dogs", bench="dow")
    assert s["mean_excess"] > 0
    assert t > 2, f"planted small-dogs t = {t:.2f} — harness failed to bank planted signal"


def test_spine_planted_small_beats_dogs(planted_panel):
    """With the edge planted in the low-price names, Small Dogs should beat full Dogs."""
    ann = strategy.annual_returns_from_synthetic(planted_panel)
    t = strategy.hac_tstat(ann["small_vs_dogs"].to_numpy())
    assert ann["small_vs_dogs"].mean() > 0


def test_spine_null_no_small_dogs_edge(null_panel):
    """With yield/price as noise the engine MUST NOT manufacture a t > 2."""
    ann = strategy.annual_returns_from_synthetic(null_panel)
    t = strategy.hac_tstat(ann["excess_small"].to_numpy())
    assert abs(t) < 2, f"null small-dogs t = {t:.2f} — spurious edge found"


def test_synthetic_engine_returns_correct_columns(planted_panel):
    """The synthetic engine returns all expected columns."""
    ann = strategy.annual_returns_from_synthetic(planted_panel)
    for col in ("dogs", "small_dogs", "foolish4", "dow",
                "excess_dogs", "excess_small", "excess_f4",
                "small_vs_dogs", "f4_vs_dogs"):
        assert col in ann.columns, f"missing column {col}"


def test_synthetic_engine_row_count(planted_panel):
    """One row per year in the synthetic panel."""
    ann = strategy.annual_returns_from_synthetic(planted_panel)
    assert len(ann) == planted_panel["truth"]["n_years"]
