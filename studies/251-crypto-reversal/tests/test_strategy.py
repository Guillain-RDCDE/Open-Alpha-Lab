"""Cross-sectional engine invariants and the study's spine — Study 251 (Crypto-Reversal).

The spine: on a synthetic tape with a *planted* reversal, the reversal long-short must win
and the Fama-MacBeth slope must be negative (recent losers earn more); on a null tape
nothing significant appears; the shuffle control is ~0. The real-data tests (guarded) encode
the study's finding — the public panel shows *momentum*, not the paper's reversal, and even
that momentum dies after crypto turnover costs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_reversal import data, strategy as st  # noqa: E402

_CACHE_PATH = data._cache_path()
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="crypto panel cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Signal & engine invariants
# ---------------------------------------------------------------------------
def test_trailing_signal_is_lagged(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices, lookback=21, skip=1)
    # First lookback+skip rows must be NaN (no warmed-up trailing window yet).
    assert sig.iloc[:21].isna().all().all()
    assert np.isfinite(sig.iloc[100]).any()


def test_reversal_is_negative_of_momentum_gross(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices)
    rev = st.long_short_returns(prices, sig, leg="reversal", cost_bps=0.0)
    mom = st.long_short_returns(prices, sig, leg="momentum", cost_bps=0.0)
    assert np.allclose(rev.to_numpy(), -mom.to_numpy())


def test_costs_lower_returns(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices)
    means = [st.long_short_returns(prices, sig, leg="momentum", cost_bps=c).mean()
             for c in [0.0, 20.0, 100.0]]
    assert means[0] >= means[1] >= means[2]


def test_turnover_in_unit_interval(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices)
    to = st.avg_turnover(sig, leg="reversal")
    assert 0.0 <= to <= 1.0


def test_invalid_leg_raises(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices)
    with pytest.raises(ValueError):
        st.long_short_returns(prices, sig, leg="banana")


def test_summary_keys_present(planted):
    prices, _ = planted
    sig = st.trailing_signal(prices)
    s = st.summary(st.long_short_returns(prices, sig, leg="reversal"))
    for k in ["n_days", "ann_return", "cagr", "sharpe", "vol_ann",
              "max_drawdown", "mean_daily_bps", "tstat"]:
        assert k in s


# ---------------------------------------------------------------------------
# The spine — synthetic positive/negative controls
# ---------------------------------------------------------------------------
def test_planted_reversal_is_recovered(planted):
    """With a planted reversal, the reversal arm beats momentum and FM slope is negative."""
    prices, _ = planted
    res = st.compare_arms(prices, cost_bps=0.0)
    assert res["reversal"]["sharpe"] > res["momentum"]["sharpe"]
    assert res["reversal"]["sharpe"] > 0.0
    assert res["reversal"]["tstat"] > 0.0
    # Negative slope == recent losers earn more == reversal.
    assert res["fama_macbeth"]["tstat_nw"] < 0.0
    assert res["fama_macbeth"]["reversal_premium_ann"] > 0.0


def test_null_panel_has_no_significant_spread(null_panel):
    """On the null tape, the reversal arm should not clear the inference bar."""
    prices, _ = null_panel
    res = st.compare_arms(prices, cost_bps=0.0)
    assert abs(res["reversal"]["tstat"]) < 2.0
    assert abs(res["fama_macbeth"]["tstat_nw"]) < 2.0


def test_shuffle_control_is_near_zero(planted):
    """Shuffling the cross-section destroys the spread even when a real signal exists."""
    prices, _ = planted
    sig = st.trailing_signal(prices)
    shuf = st.long_short_returns(prices, st.shuffle_signal(sig, seed=42), leg="reversal")
    assert abs(st.summary(shuf)["tstat"]) < 2.0


# ---------------------------------------------------------------------------
# Real-data tests (guarded) — the study's finding
# ---------------------------------------------------------------------------
@requires_cache
def test_real_panel_shows_momentum_not_reversal():
    """On the public survivor panel the paper's reversal flips sign to momentum."""
    panel = data.fetch_panel(fetch=False)
    res = st.compare_arms(panel, lookback=21, skip=1, cost_bps=0.0)
    # Reversal loses (negative t); momentum wins. FM slope positive == winners win.
    assert res["reversal"]["tstat"] < 0.0
    assert res["momentum"]["sharpe"] > res["reversal"]["sharpe"]
    assert res["fama_macbeth"]["tstat_nw"] > 0.0           # momentum sign
    assert res["fama_macbeth"]["reversal_premium_ann"] < 0.0


@requires_cache
def test_real_momentum_dies_after_costs():
    """Even the profitable momentum side is a mirage at the paper's >100 bps crypto spread."""
    panel = data.fetch_panel(fetch=False)
    sig = st.trailing_signal(panel, lookback=21, skip=1)
    gross = st.summary(st.long_short_returns(panel, sig, leg="momentum", cost_bps=0.0))
    net = st.summary(st.long_short_returns(panel, sig, leg="momentum", cost_bps=100.0))
    assert gross["ann_return"] > 0.0
    assert net["ann_return"] < 0.0
