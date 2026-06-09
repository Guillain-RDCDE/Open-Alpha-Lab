"""On a random walk the two-sided book fabricates a large, 'significant' edge while the one-sided book
has nothing — and the two-sided cycle correlates with future returns (the look-ahead fingerprint)."""

from crystal_ball import strategy, decompose, extension


def test_two_sided_fakes_edge_on_random_walk(null_close):
    lb = decompose.lookahead_bias(null_close, cost_bps=1.0, lam=1e6)
    assert lb["two_sided_sharpe"] > 1.0          # a glorious edge on pure noise...
    assert lb["two_sided_t"] > 4.0               # ...even "highly significant"
    assert abs(lb["one_sided_sharpe"]) < 0.4     # ...and nothing once causal
    assert lb["lookahead_sharpe_gap"] > 1.0


def test_future_leakage_only_two_sided(null_close):
    fl = decompose.future_leakage(null_close, lam=1e6)
    # the two-sided cycle correlates with the next-5-day return; the one-sided barely does
    assert abs(fl[5]["two_sided_corr"]) > 0.2
    assert abs(fl[5]["one_sided_corr"]) < abs(fl[5]["two_sided_corr"]) / 2


def test_honest_edge_nothing_on_null(null_close):
    he = decompose.honest_edge(null_close, cost_bps=1.0, lam=1e6)
    assert abs(he["t_stat"]) < 2.0               # the causal book has no real edge on a random walk


def test_honest_edge_recovers_real_reversion(revert_close):
    he = decompose.honest_edge(revert_close, cost_bps=1.0, lam=1e6)
    assert he["sharpe"] > 0.3                     # when reversion is real, the honest filter finds some


def test_bias_survives_lam_and_cost(null_close):
    ls = extension.lam_sweep(null_close)
    cs = extension.cost_robustness(null_close)
    # at every smoothing and every cost the two-sided book stays large, the one-sided ~0
    assert (ls["two_sided_sharpe"] > 0.8).all()
    assert (cs["two_sided_sharpe"] > 0.8).all()
    assert ls["one_sided_sharpe"].abs().max() < 0.6
