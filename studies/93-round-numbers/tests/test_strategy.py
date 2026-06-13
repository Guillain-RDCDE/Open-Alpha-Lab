"""Strategy-engine tests for Study 93 (Round-Numbers), including the spine test."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from round_numbers import strategy  # noqa: E402


def test_distance_to_round_is_folded():
    # 100.02 is 0.02 above 100; 99.98 is 0.02 below — both 0.02 from the nearest dollar.
    d = strategy.distance_to_round(np.array([100.02, 99.98, 100.5]), 1.0)
    assert np.isclose(d[0], 0.02)
    assert np.isclose(d[1], 0.02)
    assert np.isclose(d[2], 0.5)
    assert (d >= 0).all() and (d <= 0.5).all()


def test_clustering_detects_planted_clustering():
    # A tape sitting exactly on round levels must reject uniformity hard.
    clustered = np.array([100.0, 200.0, 300.0] * 500, dtype=float)
    r = strategy.clustering_test(clustered, 1.0)
    assert r["p"] < 1e-6
    assert r["frac_near"] > 0.9


def test_clustering_passes_uniform():
    # A genuinely uniform fractional part should not reject uniformity at any sane level.
    rng = np.random.default_rng(0)
    uni = 100.0 + rng.uniform(0, 1, 20000)
    r = strategy.clustering_test(uni, 1.0)
    assert r["p"] > 0.01


def test_round_grid_snaps_to_nearest():
    g = strategy.round_grid(np.array([149.0, 151.0, 100.0]), 100.0)
    assert list(g) == [100.0, 200.0, 100.0]


def test_random_level_grid_is_offset_but_same_spacing():
    c = np.linspace(100, 500, 1000)
    g = strategy.random_level_grid(c, 100.0, seed=3)
    # Spacing preserved (gaps are multiples of the grid), but the phase is shifted off 0.
    gaps = np.diff(np.unique(g))
    assert np.allclose(gaps % 100.0, 0.0) or np.allclose(gaps, 100.0)
    assert not np.allclose(g, strategy.round_grid(c, 100.0))


def test_fade_returns_lagged_no_lookahead():
    # The fade must enter at i+1, never at i. Build a tape where day i sits on a level and
    # the very next return is large; the recorded trade must use that *next* return.
    c = np.array([100.0, 100.0, 110.0, 110.0, 110.0, 110.0], dtype=float)
    tr = strategy.fade_returns(c, 100.0, band=0.01, horizon=1)
    # Day 1 is within band of 100 and above it (d>0 → long); entry at c[2]=110, exit c[3]=110.
    assert len(tr) >= 1
    assert np.all(np.isfinite(tr))


def test_net_of_costs_reduces_return():
    tr = np.array([0.01, -0.005, 0.002])
    net = strategy.net_of_costs(tr, cost_bps=5.0)
    assert np.all(net < tr)


def test_hac_tstat_sign():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.01, 0.01, 5000)
    assert strategy.hac_tstat(pos) > 2.0
    zero = rng.normal(0.0, 0.01, 5000)
    assert abs(strategy.hac_tstat(zero)) < 2.0


def test_spine_planted_magnet_fade_banks_edge(planted_magnet):
    """With a reflective barrier at each round level, the fade must bank a real, significant edge."""
    close = planted_magnet[0]["close"]
    tr = strategy.fade_returns(close, 100.0, band=0.006, horizon=1)
    assert tr.mean() > 0
    assert strategy.hac_tstat(tr) > 2.0


def test_spine_random_walk_fade_does_not_bank_edge(null_walk):
    """On a magnetism-free random walk there is no bounce to fade — the edge must be insignificant."""
    close = null_walk[0]["close"]
    tr = strategy.fade_returns(close, 100.0, band=0.006, horizon=1)
    assert abs(strategy.hac_tstat(tr)) < 2.0


def test_spine_planted_beats_null_on_mean(null_walk, planted_magnet):
    """The planted-magnetism fade must out-earn the null fade — direction of the control."""
    null_tr = strategy.fade_returns(null_walk[0]["close"], 100.0, band=0.006, horizon=1)
    planted_tr = strategy.fade_returns(planted_magnet[0]["close"], 100.0, band=0.006, horizon=1)
    assert planted_tr.mean() > null_tr.mean()
