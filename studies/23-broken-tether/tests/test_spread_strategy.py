"""The half-life engine cleanly separates a cointegrated pair (short, reverting) from a spurious one
(near-random-walk), and the pairs book makes money on the real cointegration while breaking out of
sample on the spurious pair."""

import numpy as np

from broken_tether import data, spread, strategy, decompose


def test_deterministic_and_cointegrated(coint):
    px, truth = coint
    px2, _ = data.synthetic_pair(revert_rho=0.93, seed=23)
    assert np.allclose(px.to_numpy(), px2.to_numpy())
    assert truth.is_cointegrated


def test_half_life_separates_coint_from_spurious(coint_px, spurious_px):
    hl_c = spread.half_life(spread.spread(coint_px["A"], coint_px["B"]))
    hl_s = spread.half_life(spread.spread(spurious_px["A"], spurious_px["B"]))
    assert hl_c < 60                      # cointegrated: fast mean reversion
    assert hl_s > hl_c * 2                # spurious: a far longer (random-walk-like) half-life


def test_stationarity_flags(coint_px):
    st = spread.stationarity(coint_px["A"], coint_px["B"])
    assert st["is_reverting"]
    assert 0.0 < st["hedge_ratio"] < 3.0


def test_pairs_makes_money_on_cointegration(coint_px):
    cmp = strategy.compare(coint_px["A"], coint_px["B"], cost_bps=2.0)
    assert cmp["sharpe"] > 0.1
    assert cmp["trades"] > 10


def test_oos_survives_coint_breaks_spurious(coint_px, spurious_px):
    oc = decompose.in_sample_vs_oos(coint_px["A"], coint_px["B"], cost_bps=2.0)
    os_ = decompose.in_sample_vs_oos(spurious_px["A"], spurious_px["B"], cost_bps=2.0)
    assert oc["second_half_sharpe"] > os_["second_half_sharpe"]   # real pair holds up better OOS
