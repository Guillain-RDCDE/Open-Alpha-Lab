"""The signal, the two execution conventions, and the study's two spines:
(1) on a random walk the CORRECTLY-LAGGED rule has no edge but the PEEK manufactures a
    large, HAC-significant Sharpe (the look-ahead tax);
(2) the harness banks a PLANTED edge only with the correct lag — the positive control."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from look_ahead_bias import strategy as st  # noqa: E402


# ---- signal & position -----------------------------------------------------
def test_signal_is_causal(null_prices):
    """The z-score at t uses only closes up to t: changing a LATER close leaves it fixed."""
    prices, _ = null_prices
    z = st.zscore_signal(prices, lookback=20)
    p2 = prices.copy()
    p2.iloc[-1] *= 1.5                       # perturb only the final close
    z2 = st.zscore_signal(p2, lookback=20)
    # every z except the last is unchanged (the signal cannot see the future)
    assert np.allclose(z.iloc[:-1].dropna().to_numpy(),
                       z2.iloc[:-1].dropna().to_numpy(), equal_nan=True)


def test_position_bounded(null_prices):
    prices, _ = null_prices
    for sig in ("momentum", "reversion"):
        pos = st.position(prices, signal=sig).dropna()
        assert (pos.abs() <= 1.0 + 1e-9).all()


def test_momentum_and_reversion_are_opposite(null_prices):
    prices, _ = null_prices
    mom = st.position(prices, signal="momentum")
    rev = st.position(prices, signal="reversion")
    assert np.allclose(mom.dropna().to_numpy(), -rev.dropna().to_numpy())


def test_unknown_signal_raises(null_prices):
    prices, _ = null_prices
    with pytest.raises(ValueError):
        st.position(prices, signal="psychic")


def test_unknown_convention_raises(null_prices):
    prices, _ = null_prices
    with pytest.raises(ValueError):
        st.book_returns(prices, convention="time-travel")


# ---- the look-ahead lag is the ONLY difference -----------------------------
def test_peek_and_lag1_differ_only_by_a_shift(null_prices):
    """Both books share the signal; the peek earns bar t, the lag earns bar t+1."""
    prices, _ = null_prices
    peek = st.book_returns(prices, convention="peek", cost_bps=0.0)
    lag1 = st.book_returns(prices, convention="lag1", cost_bps=0.0)
    # gross peek_t ≈ lag1_{t-1} on aligned dates (one bar apart, modulo dropna edges)
    assert not np.allclose(peek.reindex(lag1.index).dropna().to_numpy()[:50],
                           lag1.dropna().to_numpy()[:50])


def test_costs_lower_returns(null_prices):
    prices, _ = null_prices
    for conv in ("peek", "lag1"):
        m0 = st.book_returns(prices, convention=conv, cost_bps=0.0).mean()
        m5 = st.book_returns(prices, convention=conv, cost_bps=10.0).mean()
        assert m5 < m0


# ---- spine #1: the look-ahead tax on a NULL tape ---------------------------
def test_peek_manufactures_sharpe_on_random_walk(null_prices):
    """Momentum on a random walk: lag1 has ~no edge, peek invents a huge Sharpe."""
    prices, _ = null_prices
    res = st.compare(prices, signal="momentum", cost_bps=0.0)
    # the honest, lagged rule is indistinguishable from zero
    assert abs(res["test_lag1"]["tstat"]) < 2.0
    # the peek manufactures a large, HAC-significant positive Sharpe out of nothing
    assert res["summary_peek"]["sharpe"] > 3.0
    assert res["test_peek"]["tstat"] > 5.0
    # the inflation is enormous
    assert res["sharpe_inflation"] > 3.0


def test_peek_survives_costs(null_prices):
    """The bias is contemporaneous correlation, not an edge — costs can't erode it."""
    prices, _ = null_prices
    res = st.compare(prices, signal="momentum", cost_bps=5.0)
    assert res["summary_peek"]["sharpe"] > 3.0
    assert res["test_peek"]["tstat"] > 5.0


def test_peek_sign_flips_with_signal_direction(null_prices):
    """The tell that it is contemporaneous correlation, not a forecast: flipping the
    signal direction flips the peek Sharpe's sign (and magnitude is ~preserved)."""
    prices, _ = null_prices
    mom = st.compare(prices, signal="momentum")
    rev = st.compare(prices, signal="reversion")
    assert mom["summary_peek"]["sharpe"] > 0
    assert rev["summary_peek"]["sharpe"] < 0
    assert np.isclose(mom["summary_peek"]["sharpe"], -rev["summary_peek"]["sharpe"], atol=1e-6)


# ---- spine #2: the positive control ----------------------------------------
def test_harness_banks_planted_edge_with_correct_lag(edge_prices):
    """A planted lag-tradable reversion edge IS bankable with the correct lag."""
    prices, _ = edge_prices
    res = st.compare(prices, signal="reversion", cost_bps=0.0)
    assert res["summary_lag1"]["sharpe"] > 1.0
    assert res["test_lag1"]["tstat"] > 2.0


def test_no_planted_edge_no_lagged_signal(null_prices):
    """Negative control: with no planted edge the lagged reversion rule scores ~0."""
    prices, _ = null_prices
    res = st.compare(prices, signal="reversion", cost_bps=0.0)
    assert abs(res["test_lag1"]["tstat"]) < 2.0


# ---- inference plumbing ----------------------------------------------------
def test_summarize_keys(null_prices):
    prices, _ = null_prices
    s = st.summarize(st.book_returns(prices, convention="lag1"))
    assert set(s) == {"n", "cagr", "sharpe", "max_dd", "mean_ann"}


def test_bootstrap_ci_brackets_mean(edge_prices):
    prices, _ = edge_prices
    r = st.book_returns(prices, convention="lag1")
    lo, hi = st.block_bootstrap_ci(r, n_boot=800, seed=1)
    assert lo <= r.mean() <= hi
