"""Strategy invariants, the random-day control, and the study's spine:
the calendar signal beats a random-day placebo only when the effect is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from window_dressing import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for real-data tests
# ---------------------------------------------------------------------------
_SPY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "SPY_total_return.parquet"
)
_GSPC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "^GSPC_split_only.parquet"
)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_SPY_PATH) or os.path.exists(_GSPC_PATH)),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# HAC t-stat internals
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    out = st._hac_tstat(r)
    assert abs(out["tstat"]) < 3.0  # should not reject the null for white noise


def test_hac_tstat_large_signal():
    rng = np.random.default_rng(1)
    # Add a very large signal-to-noise ratio: mean=0.01, tiny noise.
    r = 0.01 + rng.normal(0, 0.0001, 500)
    out = st._hac_tstat(r)
    assert out["tstat"] > 10.0


def test_hac_tstat_small_n():
    r = np.array([0.01, 0.02])
    out = st._hac_tstat(r)
    assert np.isnan(out["tstat"])


# ---------------------------------------------------------------------------
# window_comparison
# ---------------------------------------------------------------------------
def test_window_comparison_output_keys(null_tape):
    df, _ = null_tape
    out = st.window_comparison(df, "is_pump")
    for key in ("n_pump", "mean_pump_bps", "tstat_pump", "contrast_bps", "pval_welch"):
        assert key in out


def test_window_comparison_finds_signal_when_planted(effect_tape):
    """With a strong effect, the pump contrast is clearly positive and the reversal negative."""
    df, _ = effect_tape
    pump = st.window_comparison(df, "is_pump")
    rev = st.window_comparison(df, "is_reversal")
    assert pump["contrast_bps"] > 0, "pump should earn more than baseline"
    assert rev["contrast_bps"] < 0, "reversal should earn less than baseline"


def test_window_comparison_no_signal_on_null(null_tape):
    """On a null tape the pump contrast is not statistically significant."""
    df, _ = null_tape
    pump = st.window_comparison(df, "is_pump")
    # With no planted signal, |t| should stay well below 2 (with high probability).
    assert abs(pump["tstat_pump"]) < 3.0


# ---------------------------------------------------------------------------
# pump_minus_reversal
# ---------------------------------------------------------------------------
def test_pump_minus_reversal_columns(null_tape):
    df, _ = null_tape
    out = st.pump_minus_reversal(df)
    for key in ("n_pump", "n_reversal", "mean_spread_bps", "tstat", "ann_ret"):
        assert key in out


def test_pump_minus_reversal_positive_when_effect_planted(effect_tape):
    df, _ = effect_tape
    out = st.pump_minus_reversal(df)
    assert out["mean_spread_bps"] > 0
    assert out["tstat"] > 2.0


def test_pump_minus_reversal_near_zero_on_null(null_tape):
    df, _ = null_tape
    out = st.pump_minus_reversal(df)
    assert abs(out["mean_spread_bps"]) < 5.0  # no planted signal → spread ≈ 0


# ---------------------------------------------------------------------------
# random_day_control
# ---------------------------------------------------------------------------
def test_random_day_control_reproducible(null_tape):
    df, truth = null_tape
    a = st.random_day_control(df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=1)
    b = st.random_day_control(df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=1)
    assert a["mean_bps"] == b["mean_bps"]


def test_random_day_control_different_seeds(null_tape):
    df, truth = null_tape
    a = st.random_day_control(df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=1)
    b = st.random_day_control(df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=2)
    assert a["mean_bps"] != b["mean_bps"]


def test_signal_beats_control_only_with_effect():
    """The spine: with a planted WD effect the calendar beats random-day timing;
    on a null tape it does not (calendar is a fair coin)."""
    def edge(df, truth):
        pmr = st.pump_minus_reversal(df)
        ctrl = st.random_day_control(df, n_pump=truth["n_pump"], n_reversal=truth["n_reversal"], seed=0)
        return pmr["mean_spread_bps"] - ctrl["mean_bps"]

    df_eff, truth_eff = data.synthetic_daily(n_years=50, wd_effect=1.5, seed=193)
    df_null, truth_null = data.synthetic_daily(n_years=50, wd_effect=0.0, seed=193)

    assert edge(df_eff, truth_eff) > 0, "calendar should beat random-day when effect is planted"
    # On a null tape the calendar should not consistently beat the coin.
    # We allow a generous margin: the point is not that the null always loses,
    # but that there is no reliable outperformance across seeds.
    null_edges = [
        edge(data.synthetic_daily(n_years=50, wd_effect=0.0, seed=s)[0],
             data.synthetic_daily(n_years=50, wd_effect=0.0, seed=s)[1])
        for s in range(10)
    ]
    # Over 10 seeds the mean edge on a null tape is not robustly positive.
    assert np.mean(null_edges) < 1.5


# ---------------------------------------------------------------------------
# window_sweep
# ---------------------------------------------------------------------------
def test_window_sweep_shape(null_tape):
    df, _ = null_tape
    sw = st.window_sweep(df, max_window=5)
    assert len(sw) == 5
    assert "contrast_pump_bps" in sw.columns
    assert "pval_bonferroni" in sw.columns


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    df, _ = null_tape
    out = st.summarize(df)
    for key in (
        "n_pump", "mean_pump_bps", "tstat_pump", "pval_pump",
        "n_reversal", "mean_reversal_bps", "tstat_reversal", "pval_reversal",
        "mean_spread_bps", "tstat_spread", "ann_ret",
        "ctrl_mean_bps", "ctrl_tstat",
    ):
        assert key in out, f"missing key: {key}"


# ---------------------------------------------------------------------------
# Real-data tests (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_summarize_runs():
    df = data.fetch_spy()
    out = st.summarize(df)
    assert "tstat_spread" in out
    assert np.isfinite(out["tstat_spread"])


@requires_cache
def test_real_pump_count():
    df = data.fetch_spy()
    out = st.summarize(df)
    # SPY since 1993: ~33 years × 4 × 5 ≈ 660 pump days.
    assert 500 <= out["n_pump"] <= 800


@requires_cache
def test_real_window_sweep_runs():
    df = data.fetch_spy()
    sw = st.window_sweep(df, max_window=10)
    assert len(sw) == 10
    assert sw["pval_bonferroni"].notna().all()
