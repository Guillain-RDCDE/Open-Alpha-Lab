"""CMO indicator, the two signal framings, the forward-return engine, the random control,
and the study's spine: each CMO framing beats a coin only when the matching structure is
planted in the tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chande_momentum import strategy as st  # noqa: E402
from chande_momentum import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: real-data tests must be skipped if the study cache is absent.
# ---------------------------------------------------------------------------
_STUDY_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = os.path.join(_STUDY_CACHE, "bars_SPY_1d.parquet")
requires_cache = pytest.mark.skipif(
    not os.path.exists(_SPY_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---- CMO indicator ---------------------------------------------------------
def test_cmo_range_and_nans():
    """CMO must lie in [-100, +100] and be NaN for the first ``period`` bars."""
    bars, _ = data.synthetic_daily(n_days=200, seed=185)
    c = st.cmo(bars["close"], period=14)
    valid = c.dropna()
    assert len(c[c.isna()]) == 14, "first 14 bars must be NaN (warmup)"
    assert (valid >= -100.0 - 1e-9).all()
    assert (valid <= 100.0 + 1e-9).all()


def test_cmo_all_up_is_100():
    """A uniformly rising close series must produce CMO = +100."""
    close = pd.Series(np.arange(1.0, 30.0), name="close")
    c = st.cmo(close, period=14)
    assert np.allclose(c.dropna().to_numpy(), 100.0), "all-up bars → CMO = +100"


def test_cmo_all_down_is_minus100():
    """A uniformly falling close series must produce CMO = -100."""
    close = pd.Series(np.arange(30.0, 1.0, -1.0), name="close")
    c = st.cmo(close, period=14)
    assert np.allclose(c.dropna().to_numpy(), -100.0), "all-down bars → CMO = -100"


def test_cmo_symmetric():
    """Reversing the price series should flip the sign of CMO."""
    close = pd.Series([100, 102, 101, 103, 99, 98, 100, 101, 102, 104, 103, 105, 106, 104, 105])
    c_fwd = st.cmo(close, period=14)
    c_rev = st.cmo(close[::-1].reset_index(drop=True), period=14)
    valid_fwd = c_fwd.dropna()
    valid_rev = c_rev.dropna()
    assert len(valid_fwd) > 0
    # Reversed series flips all deltas → Su and Sd swap → CMO flips sign.
    assert np.allclose(valid_fwd.to_numpy(), -valid_rev.to_numpy(), atol=1e-6)


def test_cmo_deterministic(coin):
    bars, _ = coin
    c1 = st.cmo(bars["close"], period=14)
    c2 = st.cmo(bars["close"], period=14)
    assert np.allclose(c1.dropna().to_numpy(), c2.dropna().to_numpy())


# ---- ob_os_entries ---------------------------------------------------------
def test_ob_os_entries_directions(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.ob_os_entries(c, threshold=50.0)
    assert set(ent["dir"].unique()) <= {-1, 1}
    # With 500 days there must be some entries in both directions.
    assert (ent["dir"] == 1).sum() >= 1
    assert (ent["dir"] == -1).sum() >= 1


def test_ob_os_entries_no_lookahead(coin):
    """Signals must only appear at or after bar period-1 (CMO has NaN warmup)."""
    bars, _ = coin
    c = st.cmo(bars["close"], period=14)
    ent = st.ob_os_entries(c)
    first_valid_idx = c[c.notna()].index[0]
    assert ent.index.min() >= first_valid_idx


def test_ob_os_high_threshold_fewer_entries(coin):
    """Tighter threshold should produce fewer entries (not more)."""
    bars, _ = coin
    c = st.cmo(bars["close"])
    n_50 = len(st.ob_os_entries(c, threshold=50.0))
    n_70 = len(st.ob_os_entries(c, threshold=70.0))
    assert n_50 >= n_70


# ---- zero_cross_entries ----------------------------------------------------
def test_zero_cross_entries_directions(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.zero_cross_entries(c)
    assert set(ent["dir"].unique()) <= {-1, 1}
    assert (ent["dir"] == 1).sum() >= 1
    assert (ent["dir"] == -1).sum() >= 1


def test_zero_cross_monotone_series_has_no_crosses():
    """A uniformly rising series stays CMO > 0 after warmup — no zero crosses."""
    bars, _ = data.synthetic_daily(n_days=200, momentum=0.0, seed=1)
    # Force a strictly monotone close.
    bars["close"] = np.arange(1.0, len(bars) + 1.0)
    c = st.cmo(bars["close"])
    ent = st.zero_cross_entries(c)
    assert len(ent) == 0


# ---- run_trades engine invariants ------------------------------------------
def test_ledger_columns(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.ob_os_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net",
    ]


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.ob_os_entries(c)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_bars_held_bounded(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.ob_os_entries(c)
    led = st.run_trades(bars, ent, hold_days=10, cost_bps=0.0)
    assert (led["bars_held"] >= 1).all()
    assert (led["bars_held"] <= 10).all()


def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(2000, seed=5)
    b = st.random_directions(2000, seed=5)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.40 < (a == 1).mean() < 0.60


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    c = st.cmo(bars["close"])
    ent = st.ob_os_entries(c)
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=cost))["mean_bps"]
        for cost in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- the random control & the study's spine --------------------------------
def _avg_edge(n_days, mean_rev, momentum, seeds, framing="ob_os", seed_ctrl=1):
    """Average (signal - coin) across multiple tape seeds; robust to single-seed noise."""
    deltas = []
    for seed in seeds:
        bars, _ = data.synthetic_daily(n_days=n_days, mean_rev=mean_rev,
                                       momentum=momentum, seed=seed)
        c = st.cmo(bars["close"])
        ent = st.ob_os_entries(c) if framing == "ob_os" else st.zero_cross_entries(c)
        if len(ent) < 5:
            continue
        sig = st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross")
        rnd = st.summarize(
            st.run_trades(bars, ent, hold_days=5, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=seed_ctrl)),
            "ret_gross",
        )
        deltas.append(sig["mean_bps"] - rnd["mean_bps"])
    return float(np.mean(deltas)) if deltas else float("nan")


_SEEDS = [185, 42, 99, 7, 1234, 5678]


def test_ob_os_beats_coin_only_with_mean_reversion():
    """OB/OS framing: with planted mean-reversion the signal edges a coin on average
    across seeds; on a martingale the average edge is negative (no structure to harvest).

    We average over 6 tape seeds so the test is robust to single-seed Monte Carlo noise
    while still being deterministic and offline."""
    # Planted mean-reversion → average signal-minus-coin positive.
    avg_mr = _avg_edge(1000, mean_rev=0.40, momentum=0.0, seeds=_SEEDS, framing="ob_os")
    # Martingale → average signal-minus-coin not reliably positive.
    avg_flat = _avg_edge(1000, mean_rev=0.00, momentum=0.0, seeds=_SEEDS, framing="ob_os")
    assert avg_mr > 0.0, f"expected OB/OS edge on mean-reverting tape, got {avg_mr:.2f}"
    assert avg_mr > avg_flat, (
        f"OB/OS edge should be stronger under mean-reversion ({avg_mr:.2f}) "
        f"than on a martingale ({avg_flat:.2f})"
    )


def test_zero_cross_beats_coin_only_with_momentum():
    """Zero-cross framing: with planted momentum the signal edges a coin on average
    across seeds; on a martingale it does not reliably beat a fair coin."""
    avg_mom = _avg_edge(1000, mean_rev=0.0, momentum=0.40, seeds=_SEEDS, framing="zc",
                        seed_ctrl=2)
    avg_flat = _avg_edge(1000, mean_rev=0.0, momentum=0.00, seeds=_SEEDS, framing="zc",
                         seed_ctrl=2)
    assert avg_mom > 0.0, f"expected zero-cross edge on trending tape, got {avg_mom:.2f}"
    assert avg_mom > avg_flat, (
        f"zero-cross edge should be stronger under momentum ({avg_mom:.2f}) "
        f"than on a martingale ({avg_flat:.2f})"
    )


# ---- real-data (cache-gated) -----------------------------------------------
@requires_cache
def test_real_tape_loads_and_cmo_is_finite():
    """If the cache exists: SPY daily tape loads cleanly and CMO has finite values."""
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=_STUDY_CACHE)
    assert len(bars) > 100
    c = st.cmo(bars["close"])
    assert c.dropna().notna().all()
    assert (c.dropna().abs() <= 100 + 1e-9).all()
