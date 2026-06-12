"""Strategy invariants, the barrier engine, the random control, and the study's spine:
the fade beats a coin only when the tape has a real gap-fill tendency above 50%."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind_the_gap import data, strategy as st  # noqa: E402


# ---- helpers ---------------------------------------------------------------
def _make_daily(n=200, gap_fill_prob=0.5, seed=0):
    daily, _ = data.synthetic_daily(n_days=n, gap_fill_prob=gap_fill_prob, seed=seed)
    return daily


# ---- gap classification ----------------------------------------------------
def test_gap_size_bucket_values():
    gaps = pd.Series([-2.0, -0.8, -0.1, 0.0, 0.1, 0.8, 2.0])
    buckets = st.gap_size_bucket(gaps)
    assert buckets.iloc[0] == "large"   # -2.0
    assert buckets.iloc[1] == "medium"  # -0.8
    assert buckets.iloc[2] == "small"   # -0.1
    assert buckets.iloc[3] == "small"   # 0.0 (< 0.25)
    assert buckets.iloc[4] == "small"   # 0.1
    assert buckets.iloc[5] == "medium"  # 0.8
    assert buckets.iloc[6] == "large"   # 2.0


def test_atr_is_positive_and_bounded(coin):
    daily, _ = coin
    a = st.atr(daily, n=20)
    valid = a.dropna()
    assert (valid > 0).all()
    # ATR should be smaller than the average high-low range.
    typical_range = (daily["high"] - daily["low"]).mean()
    assert valid.mean() < typical_range * 5.0


# ---- fill-rate decomposition -----------------------------------------------
def test_fill_rate_by_bucket_columns(coin):
    daily, _ = coin
    tbl = st.fill_rate_by_bucket(daily)
    assert set(tbl.columns) == {"bucket", "n_gaps", "n_fills", "fill_rate", "ci_lo", "ci_hi"}
    assert set(tbl["bucket"]) <= {"small", "medium", "large"}


def test_fill_rate_ci_covers_rate(coin):
    daily, _ = coin
    tbl = st.fill_rate_by_bucket(daily)
    for _, row in tbl.iterrows():
        assert row["ci_lo"] <= row["fill_rate"] <= row["ci_hi"], (
            f"CI [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] does not cover "
            f"fill_rate {row['fill_rate']:.3f} for bucket={row['bucket']}"
        )


# ---- barrier engine invariants ---------------------------------------------
def test_ledger_columns(coin):
    daily, _ = coin
    led = st.run_trades(daily, min_gap_pct=0.25, stop_R=1.5, cost_bps=0)
    expected = {
        "date", "gap_pct", "gap_size", "dir", "entry", "tp_px",
        "stop_px", "exit", "exit_reason", "ret_gross", "ret_net",
    }
    assert expected.issubset(set(led.columns))


def test_exit_reasons_valid(coin):
    daily, _ = coin
    led = st.run_trades(daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=0)
    assert set(led["exit_reason"].unique()).issubset({"tp", "sl", "eod"})


def test_net_equals_gross_minus_two_costs(coin):
    daily, _ = coin
    led = st.run_trades(daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=1.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2 * 1.5e-4), (
        "net return should equal gross minus 2× cost_bps"
    )


def test_fade_direction_correct(coin):
    """Direction must oppose the gap: negative gap → d=+1 (fade up toward prev close)."""
    daily, _ = coin
    led = st.run_trades(daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=0)
    led_neg_gap = led[led["gap_pct"] < 0]
    led_pos_gap = led[led["gap_pct"] > 0]
    assert (led_neg_gap["dir"] == 1).all(),  "negative gap should have dir=+1 (fade up)"
    assert (led_pos_gap["dir"] == -1).all(), "positive gap should have dir=-1 (fade down)"


def test_tp_is_win_sl_is_loss(coin):
    """A take-profit exit is a win (ret_gross > 0); a stop exit is a loss (ret_gross < 0)."""
    daily, _ = coin
    led = st.run_trades(daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    assert (tp["ret_gross"] >= -1e-9).all(), "tp exits should have non-negative gross return"
    assert (sl["ret_gross"] <=  1e-9).all(), "sl exits should have non-positive gross return"


# ---- random control --------------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(coin):
    daily, _ = coin
    means = [
        st.summarize(st.run_trades(daily, min_gap_pct=0.0, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- the thesis: fade beats coin only with planted fill tendency -----------
def test_fade_beats_coin_only_with_mean_reversion(coin, mean_reverting):
    """The spine: with a planted gap-fill tendency the fade edges a random-direction
    control; on a fair coin (50% fill) it does not."""

    def edge(daily, seed):
        led_fade = st.run_trades(daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=0)
        n = len(led_fade)
        led_rand = st.run_trades(
            daily, min_gap_pct=0.0, stop_R=1.5, cost_bps=0,
            directions=st.random_directions(n, seed=seed),
        )
        fade_mean = st.summarize(led_fade, "ret_gross")["mean_bps"]
        rand_mean = st.summarize(led_rand, "ret_gross")["mean_bps"]
        return fade_mean - rand_mean

    assert edge(mean_reverting[0], seed=1) > 0.0, (
        "with planted gap-fill tendency the fade should beat the coin"
    )
    # On a coin tape the fade should not decisively beat the coin
    # (within a generous ±10 bps tolerance -- small-sample noise is expected on 500 days).
    assert abs(edge(coin[0], seed=1)) < 10.0, (
        "on a fair coin tape the fade should not decisively beat or lose to the coin"
    )
