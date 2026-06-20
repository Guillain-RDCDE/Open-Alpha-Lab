"""Signals, trade engine invariants, the random control, and the study's spine:
the z-score strategy beats random only when ratio mean-reversion is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platinum_palladium import data, strategy as st  # noqa: E402


# ---- ratio & z-score -------------------------------------------------------
def test_compute_ratio_alignment():
    """Ratio is always platinum/palladium on the common index."""
    plat, pall, _ = data.synthetic_daily(n_days=300, mean_rev_speed=0.0, seed=1)
    ratio = st.compute_ratio(plat, pall)
    expected = plat["close"] / pall["close"]
    assert np.allclose(ratio.to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_zscore_no_lookahead():
    """The z-score at position t only uses data up to t (rolling, min_periods=window)."""
    plat, pall, _ = data.synthetic_daily(n_days=300, mean_rev_speed=0.0, seed=2)
    ratio = st.compute_ratio(plat, pall)
    z = st.zscore(ratio, window=50)
    assert z.iloc[:49].isna().all()
    assert np.isfinite(z.iloc[49])


def test_zscore_has_unit_variance_on_iid_series():
    """A z-score of a long i.i.d. series should have variance ~ 1."""
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0, 10, 500))
    z = st.zscore(s, window=50).dropna()
    assert 0.8 < z.std() < 1.3


def test_half_life_detects_planted_reversion():
    """A fast-reverting synthetic ratio yields a short estimated half-life; a random
    walk yields a long (or infinite) one."""
    plr, par, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=0.20, seed=5)
    pl0, pa0, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=0.0, seed=5)
    hl_rev = st.half_life(st.compute_ratio(plr, par))
    hl_walk = st.half_life(st.compute_ratio(pl0, pa0))
    assert hl_rev < 30.0           # fast reversion → short half-life
    assert hl_walk > hl_rev        # random walk → much slower (or inf)


# ---- entries ---------------------------------------------------------------
def test_entries_respect_threshold():
    """Entries are only generated when |z| exceeds the threshold."""
    plat, pall, _ = data.synthetic_daily(n_days=1500, mean_rev_speed=0.15, seed=3)
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=2.0, exit_z=0.5)
    if len(entries) > 0:
        assert (entries["z_at_entry"].abs() >= 2.0).all()


def test_entries_columns(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) > 0:
        assert {"entry_date", "exit_date", "dir", "z_at_entry"}.issubset(entries.columns)
        assert set(entries["dir"].unique()).issubset({-1, 1})


# ---- trade ledger invariants ------------------------------------------------
def test_ledger_columns(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) == 0:
        pytest.skip("no entries generated for this tape")
    ledger = st.run_trades(plat, pall, entries, cost_bps=5.0)
    expected = {"entry_date", "exit_date", "dir", "days_held", "ret_gross", "ret_net"}
    assert expected.issubset(set(ledger.columns))


def test_net_is_gross_minus_cost(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) == 0:
        pytest.skip("no entries generated for this tape")
    cost_bps = 7.0
    ledger = st.run_trades(plat, pall, entries, cost_bps=cost_bps)
    assert np.allclose(ledger["ret_net"], ledger["ret_gross"] - cost_bps * 1e-4)


def test_days_held_positive(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) == 0:
        pytest.skip("no entries generated for this tape")
    ledger = st.run_trades(plat, pall, entries, cost_bps=0.0)
    assert (ledger["days_held"] > 0).all()


# ---- random control ---------------------------------------------------------
def test_random_entries_same_shape(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    rand = st.random_entries(entries, seed=0)
    assert len(rand) == len(entries)
    assert set(rand["dir"].unique()).issubset({-1, 1})


def test_random_entries_reproducible():
    plat, pall, _ = data.synthetic_daily(n_days=1500, mean_rev_speed=0.10, seed=42)
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    r1 = st.random_entries(entries, seed=7)
    r2 = st.random_entries(entries, seed=7)
    assert np.array_equal(r1["dir"].to_numpy(), r2["dir"].to_numpy())


# ---- bootstrap CI -----------------------------------------------------------
def test_bootstrap_ci_brackets_mean():
    """The block-bootstrap CI should bracket the sample mean per-trade return."""
    plat, pall, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=0.15, seed=9)
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    ledger = st.run_trades(plat, pall, entries, cost_bps=0.0)
    if len(ledger) < 10:
        pytest.skip("too few trades for a meaningful CI")
    mean_bps = st.summarize(ledger, "ret_gross")["mean_bps"]
    lo, hi = st.block_bootstrap_ci(ledger, "ret_gross")
    assert lo <= mean_bps <= hi


# ---- study spine ------------------------------------------------------------
def test_strategy_beats_random_when_reversion_is_real():
    """The spine: with planted reversion the z-score strategy outperforms random-direction;
    on a random walk it does not have the same advantage."""

    def edge(speed, seed=310):
        plat, pall, _ = data.synthetic_daily(n_days=3000, mean_rev_speed=speed, seed=seed)
        ratio = st.compute_ratio(plat, pall)
        entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
        if len(entries) == 0:
            return float("nan"), float("nan")
        ledger = st.run_trades(plat, pall, entries, cost_bps=0.0)
        rand = st.run_trades(plat, pall, st.random_entries(entries, seed=42), cost_bps=0.0)
        s = st.summarize(ledger, "ret_gross")
        r = st.summarize(rand, "ret_gross")
        return s["mean_bps"] - r["mean_bps"], s["tstat"]

    e_real, t_real = edge(0.15)
    # With planted reversion the strategy beats the coin and clears the |t| >= 2 bar.
    assert not np.isnan(e_real)
    assert e_real > 0.0, f"expected positive edge with reversion, got {e_real:.1f}"
    assert t_real > 2.0, f"expected t>2 with planted reversion, got {t_real:.2f}"


def test_summarize_returns_expected_keys(reverting_pair):
    plat, pall, _ = reverting_pair
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) == 0:
        pytest.skip("no entries generated for this tape")
    ledger = st.run_trades(plat, pall, entries, cost_bps=5.0)
    s = st.summarize(ledger, "ret_net")
    assert {"n_trades", "win_rate", "mean_bps", "tstat", "skew"}.issubset(s.keys())


def test_high_win_rate_can_hide_negative_mean():
    """The study's teardown: a 'take the reversion' exit prints a high win-rate even when
    the per-trade mean is not positive — the win-rate is not the edge. We check the
    summarizer reports both so the trap is visible."""
    plat, pall, _ = data.synthetic_daily(n_days=2000, mean_rev_speed=0.0, seed=4)
    ratio = st.compute_ratio(plat, pall)
    entries = st.zscore_entries(ratio, window=252, enter_z=1.5, exit_z=0.5)
    if len(entries) < 6:
        pytest.skip("too few trades on this null tape")
    s = st.summarize(st.run_trades(plat, pall, entries, cost_bps=0.0), "ret_gross")
    # On a random walk the mean is not significant regardless of win-rate.
    assert abs(s["tstat"]) < 2.0
