"""The fade engine's invariants, the random control, and the study's two spines:
(1) fading beats a coin ONLY when the pundit carries a genuine anti-signal, and
(2) pure curation of a coin-flip pundit's worst calls manufactures a significant-looking
fade with no real predictability (the selection-bias trap the third axis is about)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_cramer import data  # noqa: E402
from inverse_cramer import strategy as st  # noqa: E402


# ---- forward-return primitive ---------------------------------------------
def test_forward_return_one_lag_no_lookahead():
    """forward_return enters entry_lag bars after the call and measures fwd sessions."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    close = pd.Series(np.linspace(100, 160, 60), index=idx)  # +1/day, monotone
    call = idx[10]
    fr = st.forward_return(close, call, fwd=5, entry_lag=1)
    # entry at idx[11], exit at idx[16]
    expected = close.iloc[16] / close.iloc[11] - 1.0
    assert abs(fr - expected) < 1e-12


def test_forward_return_returns_none_off_the_edge():
    idx = pd.bdate_range("2020-01-01", periods=20)
    close = pd.Series(np.arange(1, 21, dtype=float), index=idx)
    assert st.forward_return(close, idx[18], fwd=21, entry_lag=1) is None


# ---- real ledger builder ---------------------------------------------------
def test_build_real_ledger_drops_uncovered():
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.Series(np.linspace(100, 200, 120), index=idx)
    calls = pd.DataFrame({
        "ticker": ["AAA", "BBB"],          # BBB has no tape -> dropped
        "date": [idx[10], idx[10]],
        "direction": [1, -1],
    })
    led = st.build_real_ledger(calls, {"AAA": close}, fwd=21)
    assert list(led.columns) == ["ticker", "date", "direction", "fwd_ret"]
    assert len(led) == 1 and led.iloc[0]["ticker"] == "AAA"


# ---- the fade engine -------------------------------------------------------
def test_fade_is_opposite_of_direction():
    led = pd.DataFrame({"direction": [1, -1, 1], "fwd_ret": [0.10, 0.10, -0.05]})
    r = st.fade_returns(led, cost_bps=0.0)
    # fade = -direction * fwd_ret
    assert np.allclose(r, [-0.10, 0.10, 0.05])


def test_cost_charged_both_legs():
    led = pd.DataFrame({"direction": [1], "fwd_ret": [0.10]})
    gross = st.fade_returns(led, cost_bps=0.0)[0]
    net = st.fade_returns(led, cost_bps=5.0)[0]
    assert np.isclose(gross - net, 2 * 5.0 * 1e-4)  # two legs


def test_random_control_overrides_direction():
    led = pd.DataFrame({"direction": [1, 1], "fwd_ret": [0.10, 0.10]})
    r = st.fade_returns(led, dirs=np.array([1, -1]))
    assert np.allclose(r, [0.10, -0.10])


def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---- summarize / inference -------------------------------------------------
def test_summarize_fields_and_bounds(anti_world):
    _, calls, _ = anti_world
    led = st.synthetic_ledger(calls)
    s = st.summarize(st.fade_returns(led))
    for k in ("n", "win_rate", "mean_bps", "sharpe", "skew", "tstat", "ci_lo", "ci_hi"):
        assert k in s
    assert 0.0 <= s["win_rate"] <= 1.0
    assert s["ci_lo"] <= s["mean_bps"] <= s["ci_hi"]


# ---- the two spines --------------------------------------------------------
def test_fade_beats_coin_only_with_anti_signal(null_world, anti_world):
    """Spine #1: with a planted anti-signal the fade clears the bar and beats the coin;
    on a coin-flip pundit it does not (|t| below 2)."""
    def res(world, seed):
        _, calls, _ = world
        return st.excess_over_coin(st.synthetic_ledger(calls), seed=seed)

    anti = res(anti_world, seed=336)
    null = res(null_world, seed=336)
    assert anti["fade"]["tstat"] > 2.0 and anti["excess_bps"] > 0.0
    assert abs(null["fade"]["tstat"]) < 2.0   # a coin: not distinguishable from zero


def test_curation_manufactures_fake_edge(curated_world):
    """Spine #2 (the third axis): a coin-flip pundit, curated to his worst calls, prints a
    significant-looking fade with NO real predictive content."""
    _, calls, _ = curated_world
    s = st.summarize(st.fade_returns(st.synthetic_ledger(calls)))
    assert s["win_rate"] > 0.6        # looks like a great fade
    assert s["tstat"] > 2.0           # and clears the bar...
    # ...yet it came entirely from selection: the pundit was a coin.


def test_anti_edge_monotone_in_planted_signal():
    """The fade's edge over a coin rises with the planted anti-signal (faithful detector)."""
    edges = []
    for a in (0.0, 0.3, 0.6):
        _, calls, _ = data.synthetic_calls(anti_alpha=a, selection_bias=0.0, seed=336)
        edges.append(st.excess_over_coin(st.synthetic_ledger(calls), seed=336)["excess_bps"])
    assert edges[0] < edges[1] < edges[2]
