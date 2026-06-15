"""The Fisher Transform indicator, crossover signals, the monotonicity proof, the
random control, and the study's spine: the signal adds nothing over the raw normalised
price, and on a martingale it does not beat a coin."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher_transform import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache gate
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REAL_CACHE_PATH = os.path.join(STUDY_CACHE, "bars_SPY_1d.parquet")

requires_cache = pytest.mark.skipif(
    not os.path.exists(REAL_CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Fisher Transform indicator
# ---------------------------------------------------------------------------
def test_fisher_transform_columns(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    assert set(ft.columns) == {"fisher", "trigger", "raw_x"}
    assert ft.index.equals(bars.index)


def test_fisher_transform_range_of_raw_x(coin):
    """raw_x must lie within [−clamp, +clamp] by construction."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10, clamp=0.999)
    valid = ft["raw_x"].dropna()
    assert (valid >= -0.999 - 1e-10).all()
    assert (valid <= 0.999 + 1e-10).all()


def test_fisher_is_finite_where_range_positive(coin):
    """Fisher must be finite wherever the high-low range is positive."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    valid = ft["fisher"].dropna()
    assert np.isfinite(valid.to_numpy()).all()


def test_trigger_is_fisher_lagged_by_one(coin):
    """Trigger must be Fisher shifted by exactly one bar."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    # Drop the first row where trigger is NaN from the lag.
    aligned = ft.dropna(subset=["fisher", "trigger"])
    assert np.allclose(
        aligned["trigger"].to_numpy(),
        ft["fisher"].shift(1).loc[aligned.index].to_numpy(),
    )


def test_fisher_monotone_in_raw_x(coin):
    """Fisher = atanh(raw_x) is strictly monotone: if raw_x increases, so does Fisher."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10).dropna()
    # Check pairwise: sort by raw_x, verify Fisher is also sorted (monotone).
    sorted_x = ft["raw_x"].sort_values()
    sorted_f = ft.loc[sorted_x.index, "fisher"]
    # Allow for numerical ties; require no inversions.
    diffs_x = np.diff(sorted_x.to_numpy())
    diffs_f = np.diff(sorted_f.to_numpy())
    # Where x is strictly increasing, Fisher must also be strictly increasing.
    strict = diffs_x > 1e-10
    assert (diffs_f[strict] > 0).all(), "Fisher is not monotone in raw_x — the transform changed the ordering"


# ---------------------------------------------------------------------------
# Crossover signal generation
# ---------------------------------------------------------------------------
def test_crossover_entries_have_valid_directions(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    assert set(ent["dir"].unique()) <= {-1, 1}
    assert len(ent) > 0


def test_crossover_no_signal_on_constant_price():
    """A flat price series has no range, so no Fisher crossover should fire."""
    # Constant close, but non-degenerate open/high/low (so the tape is valid).
    idx = pd.bdate_range("2020-01-02", periods=100, name="date")
    bars = pd.DataFrame({
        "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1e6,
    }, index=idx)
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    # With a flat close and a constant range, raw_x and Fisher are constant — no crosses.
    assert len(ent) == 0


def test_raw_crossover_entries_match_fisher_crossover(coin):
    """Core monotonicity claim: Fisher and raw-x crossovers fire on the same bars."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    agree = st.fisher_vs_raw_agreement(ft)
    # With clamp=0.999, the clamp is almost never active on a well-behaved tape.
    assert agree["fisher_in_raw"] > 0.99, (
        f"Fisher entries not covered by raw-x: only {agree['fisher_in_raw']:.3f}"
    )
    assert agree["raw_in_fisher"] > 0.99, (
        f"Raw-x entries not covered by Fisher: only {agree['raw_in_fisher']:.3f}"
    )


# ---------------------------------------------------------------------------
# Trade engine invariants
# ---------------------------------------------------------------------------
def test_ledger_columns_present(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit_price", "bars_held", "ret_gross", "ret_net",
    ]


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=2.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.0e-4)


def test_bars_held_is_at_most_hold_days(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=0.0)
    assert (led["bars_held"] >= 1).all()
    assert (led["bars_held"] <= 5).all()


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    means = [
        st.summarize(st.run_trades(bars, ent, hold_days=5, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


# ---------------------------------------------------------------------------
# The random control
# ---------------------------------------------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


# ---------------------------------------------------------------------------
# The study's spine — signal adds nothing over the raw value (synthetic)
# ---------------------------------------------------------------------------
def test_fisher_does_not_beat_raw_on_coin(coin):
    """On a martingale, Fisher and raw-x crossovers have statistically identical expectations."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    f_ent = st.crossover_entries(ft)
    r_ent = st.raw_crossover_entries(ft)
    f_summ = st.summarize(st.run_trades(bars, f_ent, hold_days=5, cost_bps=0), "ret_gross")
    r_summ = st.summarize(st.run_trades(bars, r_ent, hold_days=5, cost_bps=0), "ret_gross")
    # Both should be noise-level (mean close to zero).
    assert abs(f_summ["mean_bps"]) < 30.0, f"Fisher mean too large: {f_summ['mean_bps']:.2f} bps"
    assert abs(r_summ["mean_bps"]) < 30.0, f"Raw mean too large: {r_summ['mean_bps']:.2f} bps"


def test_signal_does_not_beat_coin_on_martingale(coin):
    """On a martingale, the Fisher/Trigger cross does not reliably beat a coin."""
    bars, _ = coin
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    fisher_mean = st.summarize(
        st.run_trades(bars, ent, hold_days=5, cost_bps=0), "ret_gross"
    )["mean_bps"]
    rand_mean = st.summarize(
        st.run_trades(bars, ent, hold_days=5, cost_bps=0,
                      directions=st.random_directions(len(ent), seed=42)),
        "ret_gross",
    )["mean_bps"]
    # The gap should not be large — the transform adds no information.
    assert abs(fisher_mean - rand_mean) < 25.0, (
        f"Fisher beat coin by {fisher_mean - rand_mean:.2f} bps on a martingale — unexpected"
    )


# ---------------------------------------------------------------------------
# Real-data tests — skip when cache is absent (CI offline)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_spy_fisher_transform_runs():
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    ft = st.fisher_transform(bars, period=10)
    ent = st.crossover_entries(ft)
    assert len(ent) > 50
    led = st.run_trades(bars, ent, hold_days=5, cost_bps=1.0)
    assert len(led) > 50
    summ = st.summarize(led, "ret_net")
    assert "tstat" in summ
    assert np.isfinite(summ["tstat"])


@requires_cache
def test_real_spy_monotonicity_holds():
    """On the real SPY tape, Fisher and raw-x crossovers must coincide almost perfectly."""
    bars = data.fetch_daily("SPY", fetch=False, cache_dir=STUDY_CACHE)
    ft = st.fisher_transform(bars, period=10)
    agree = st.fisher_vs_raw_agreement(ft)
    assert agree["fisher_in_raw"] > 0.99
    assert agree["raw_in_fisher"] > 0.99
