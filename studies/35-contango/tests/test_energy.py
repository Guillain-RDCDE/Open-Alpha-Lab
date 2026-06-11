"""Real-tape energy module — the contango bleed table and the roll-yield timing book.

These tests use a small **deterministic synthetic price panel** with a known, baked-in contango (the
laddered leg structurally out-returns the front), so they assert the machinery's sign and causality with no
network. A final test exercises the on-disk cache if it is present (skipped in a fresh checkout)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from contango import energy


def _synthetic_prices(n_weeks: int = 400, drag: float = 0.0015, noise: float = 0.012,
                      seed: int = 35) -> pd.DataFrame:
    """A weekly price panel for USO/USL/UNG/UNL where the front-month legs bleed a steady roll cost.

    Each laddered leg is a near-random-walk; each front leg is the same path minus a positive weekly ``drag``
    (a persistent contango), so ``laddered − front`` > 0 on average — the textbook bleed, by construction.
    ``noise`` is kept modest so the deterministic roll cost dominates the random walk in the unit tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-08", periods=n_weeks, freq="W-FRI", name="date")
    out = {}
    for front, lad in energy.PAIRS.values():
        shock = noise * rng.standard_normal(n_weeks)
        lad_ret = 0.0005 + shock
        front_ret = lad_ret - drag                      # front pays the roll every week
        out[lad] = 100 * np.exp(np.cumsum(lad_ret))
        out[front] = 100 * np.exp(np.cumsum(front_ret))
    df = pd.DataFrame(out, index=idx)
    return df[sorted(df.columns)]


def test_pairs_are_the_two_energy_curves():
    assert energy.PAIRS == {"WTI": ("USO", "USL"), "GAS": ("UNG", "UNL")}
    assert energy.ALL_TICKERS == ["UNG", "UNL", "USL", "USO"]


def test_bleed_table_detects_contango():
    """With a baked positive roll cost, the front bleeds: positive drag, >50% weeks in contango, gap>0."""
    bt = energy.bleed_table(_synthetic_prices())
    assert set(bt.index) == {"WTI", "GAS"}
    for cmd in ("WTI", "GAS"):
        assert bt.loc[cmd, "ann_drag_pct"] > 0.0                 # laddered out-earns front
        assert bt.loc[cmd, "weeks_in_contango_pct"] > 50.0       # front underperforms most weeks
        assert bt.loc[cmd, "gap_pct"] > 0.0                      # cumulative laddered − front > 0
        assert bt.loc[cmd, "front_total_pct"] < bt.loc[cmd, "lad_total_pct"]


def test_timing_book_beats_naive_long_under_contango():
    """In a persistent contango the front bleeds geometrically; shorting it (the timing book) compounds
    positively, so the book's CAGR beats buy-and-hold of the front — a noise-robust geometric invariant."""
    px = _synthetic_prices(drag=0.004)
    book = energy.book_summary(px, commodity="WTI", cost_bps=0.0)
    naive = energy.summary(energy.always_long_front(px, "WTI"))
    assert book["cagr"] > naive["cagr"]
    assert set(["sharpe", "cagr", "max_drawdown", "skew", "hac_t", "turnover_per_yr"]).issubset(book)


def test_roll_timing_book_is_causal():
    """The signal is lagged: changing only the last week's price leaves all earlier book values intact."""
    px = _synthetic_prices()
    base = energy.roll_timing_book(px, "WTI")
    bumped = px.copy()
    bumped.iloc[-1, bumped.columns.get_loc("USO")] *= 1.25       # perturb only the final front print
    after = energy.roll_timing_book(bumped, "WTI")
    n = min(len(base), len(after)) - 1
    assert np.allclose(base.to_numpy()[:n], after.to_numpy()[:n])


def test_combined_book_equal_weights_both_curves():
    px = _synthetic_prices()
    combo = energy.combined_book(px)
    assert combo.name == "combined"
    assert len(combo) > 0
    assert np.isfinite(combo.to_numpy()).all()


def test_cost_only_reduces_sharpe():
    px = _synthetic_prices()
    gross = energy.book_summary(px, commodity=None, cost_bps=0.0)["sharpe"]
    net = energy.book_summary(px, commodity=None, cost_bps=25.0)["sharpe"]
    assert net <= gross + 1e-12


def test_load_pairs_empty_on_cache_miss():
    out = energy.load_pairs(cache="/nonexistent_energy_cache_xyz.parquet")
    assert out.empty


def test_real_cache_roundtrip_if_present():
    """If the real ETF cache exists in this checkout, it loads with the four tickers and a usable history."""
    px = energy.load_pairs()
    if px.empty:
        pytest.skip("energy ETF cache absent in this checkout — run verify.py --fetch")
    assert set(energy.ALL_TICKERS).issubset(px.columns)
    bt = energy.bleed_table(px)
    # the real WTI front (USO) has historically bled to its laddered sibling — sanity, not a tight number
    assert bt.loc["WTI", "ann_drag_pct"] > 0.0
