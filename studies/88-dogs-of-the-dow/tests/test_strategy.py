"""Strategy-engine tests for Study 88 (Dogs-of-the-Dow), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dogs_of_the_dow import strategy  # noqa: E402


def test_trailing_yield_basic():
    idx = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    price = pd.Series(100.0, index=idx)
    divs = pd.Series([1.0, 1.0, 1.0, 1.0],
                     index=pd.to_datetime(["2020-03-15", "2020-06-15",
                                           "2020-09-15", "2020-12-15"]))
    # 4 dividends of $1 in the trailing year on a $100 price -> 4% yield.
    y = strategy.trailing_yield(price, divs, pd.Timestamp("2020-12-31"))
    assert abs(y - 0.04) < 1e-9


def test_trailing_yield_excludes_old_dividends():
    idx = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    price = pd.Series(50.0, index=idx)
    divs = pd.Series([1.0, 1.0],
                     index=pd.to_datetime(["2018-06-15", "2020-06-15"]))
    # Only the 2020 dividend is in the trailing 12 months ending 2020-12-31.
    y = strategy.trailing_yield(price, divs, pd.Timestamp("2020-12-31"))
    assert abs(y - (1.0 / 50.0)) < 1e-9


def test_summarize_compounds_correctly():
    ann = pd.DataFrame({"dogs": [0.10, 0.10], "dow": [0.0, 0.0],
                        "excess": [0.10, 0.10]}).assign(year=[2001, 2002]).set_index("year")
    s = strategy.summarize(ann)
    assert abs(s["dogs_cagr"] - 0.10) < 1e-9
    assert abs(s["total_dogs"] - 1.21) < 1e-9
    assert s["win_rate"] == 1.0


def test_hac_tstat_detects_strong_mean():
    x = np.full(40, 0.05) + np.random.default_rng(0).normal(0, 0.001, 40)
    assert strategy.hac_tstat(x) > 5


def test_block_bootstrap_brackets_zero_for_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.05, 30)
    lo, hi, p = strategy.block_bootstrap_ci(x, n_boot=2000)
    assert lo < 0 < hi
    assert p > 0.10


def test_alpha_beta_recovers_planted_line():
    # y = 0.02 + 1.5 x : alpha 2%, beta 1.5.
    rng = np.random.default_rng(2)
    x = rng.normal(0.08, 0.15, 200)
    y = 0.02 + 1.5 * x + rng.normal(0, 1e-6, 200)
    ann = pd.DataFrame({"dogs": y, "dow": x})
    ab = strategy.alpha_beta(ann)
    assert abs(ab["alpha"] - 0.02) < 1e-3
    assert abs(ab["beta"] - 1.5) < 1e-3


def test_spine_planted_panel_yield_beats_index(planted_panel):
    """With a planted yield->return edge, the top-yield basket MUST beat the index."""
    ann = strategy.annual_returns_from_synthetic(planted_panel)
    t = strategy.hac_tstat(ann["excess"].to_numpy())
    s = strategy.summarize(ann)
    assert s["mean_excess"] > 0
    assert t > 2  # the harness banks the planted signal


def test_spine_null_panel_yield_finds_nothing(null_panel):
    """With yield as noise, the basket must NOT clear the significance bar."""
    ann = strategy.annual_returns_from_synthetic(null_panel)
    t = strategy.hac_tstat(ann["excess"].to_numpy())
    assert abs(t) < 2  # no fake edge manufactured in the null
