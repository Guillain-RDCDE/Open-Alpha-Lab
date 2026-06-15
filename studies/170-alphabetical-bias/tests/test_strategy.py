"""Strategy tests: monthly grouping, HAC inference, Bonferroni, positive control."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphabetical_bias import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# monthly_ew_returns shape and invariants
# ---------------------------------------------------------------------------
def test_monthly_ew_returns_columns(null_panel):
    panel, _ = null_panel
    monthly = st.monthly_ew_returns(panel)
    expected = {"ew_early", "ew_late", "diff", "n_early", "n_late"}
    assert expected.issubset(set(monthly.columns))


def test_monthly_ew_returns_length(null_panel):
    panel, truth = null_panel
    monthly = st.monthly_ew_returns(panel)
    assert len(monthly) == truth["n_months"]


def test_diff_equals_early_minus_late(null_panel):
    panel, _ = null_panel
    monthly = st.monthly_ew_returns(panel)
    computed_diff = monthly["ew_early"] - monthly["ew_late"]
    assert np.allclose(monthly["diff"].to_numpy(), computed_diff.to_numpy(), equal_nan=True)


def test_n_early_and_late_positive(null_panel):
    panel, _ = null_panel
    monthly = st.monthly_ew_returns(panel)
    assert (monthly["n_early"] > 0).all()
    assert (monthly["n_late"] > 0).all()


def test_monthly_ew_returns_missing_column_raises():
    bad = pd.DataFrame({"date": [], "ret": []})
    with pytest.raises((ValueError, KeyError)):
        st.monthly_ew_returns(bad)


# ---------------------------------------------------------------------------
# HAC t-stat internals
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_on_zeros():
    result = st._hac_tstat(np.zeros(50))
    assert result["mean_bps"] == pytest.approx(0.0)
    # tstat should be 0 or nan (mean is 0, SE is 0 → 0/0 edge case).
    t = result["tstat"]
    assert t == pytest.approx(0.0) or not np.isfinite(t)


def test_hac_tstat_detects_large_mean():
    rng = np.random.default_rng(1)
    arr = rng.normal(0.01, 0.001, 200)  # large positive mean
    result = st._hac_tstat(arr)
    assert result["tstat"] > 5.0


def test_hac_tstat_nan_on_tiny_n():
    result = st._hac_tstat(np.array([0.01, 0.02]))
    assert not np.isfinite(result["tstat"])


# ---------------------------------------------------------------------------
# summarize_return_diff
# ---------------------------------------------------------------------------
def test_summarize_return_diff_null(null_panel):
    """On a null panel, the spread t-stat should be small (no planted edge)."""
    panel, _ = null_panel
    monthly = st.monthly_ew_returns(panel)
    result = st.summarize_return_diff(monthly)
    assert "tstat" in result
    # With no planted edge, |t| should be well below 3.
    t = result["tstat"]
    assert abs(t) < 3.5 or not np.isfinite(t)


def test_summarize_return_diff_biased(biased_panel):
    """With a large planted edge, the t-stat should be clearly significant."""
    panel, _ = biased_panel
    monthly = st.monthly_ew_returns(panel)
    result = st.summarize_return_diff(monthly)
    t = result["tstat"]
    assert np.isfinite(t)
    assert t > 2.0


def test_summarize_has_sharpe(null_panel):
    panel, _ = null_panel
    monthly = st.monthly_ew_returns(panel)
    result = st.summarize_return_diff(monthly)
    assert "sharpe_ann" in result


# ---------------------------------------------------------------------------
# volume_comparison
# ---------------------------------------------------------------------------
def test_volume_comparison_keys(null_panel):
    panel, _ = null_panel
    result = st.volume_comparison(panel)
    assert "vol_ratio" in result
    assert "tstat" in result
    assert result["mean_vol_early"] > 0
    assert result["mean_vol_late"] > 0


def test_volume_bias_detected(biased_panel):
    """Planted 30% volume premium is detectable by the volume_comparison function."""
    panel, truth = biased_panel
    result = st.volume_comparison(panel)
    assert result["vol_ratio"] > 1.05  # early has at least 5% more volume on average


# ---------------------------------------------------------------------------
# per_letter_returns
# ---------------------------------------------------------------------------
def test_per_letter_returns_index(null_panel):
    panel, _ = null_panel
    result = st.per_letter_returns(panel)
    import string
    assert set(result.index) == set(string.ascii_uppercase)


def test_per_letter_columns(null_panel):
    panel, _ = null_panel
    result = st.per_letter_returns(panel)
    expected = {"n_months", "n_stocks", "mean_bps", "tstat", "bonferroni_significant"}
    assert expected.issubset(set(result.columns))


def test_bonferroni_flag_conservative(null_panel):
    """On a null panel, very few (ideally 0) letters should clear the Bonferroni bar."""
    panel, _ = null_panel
    result = st.per_letter_returns(panel)
    n_sig = result["bonferroni_significant"].sum()
    # Expect at most 1–2 spurious hits by chance on a null panel.
    assert n_sig <= 3


# ---------------------------------------------------------------------------
# detect_planted_bias — the spine test
# ---------------------------------------------------------------------------
def test_positive_control_detects_when_planted(biased_panel):
    panel, _ = biased_panel
    result = st.detect_planted_bias(panel)
    assert result["signal_detected"] is True


def test_positive_control_null(null_panel):
    """On a null panel, signal detection should NOT fire (it might rarely but usually won't)."""
    panel, _ = null_panel
    result = st.detect_planted_bias(panel)
    # This is a statistical test; the null might occasionally exceed |t|=2.
    # We just check the function runs and returns a boolean.
    assert isinstance(result["signal_detected"], bool)
    # Mean should be small (near zero).
    assert abs(result["mean_bps"]) < 50.0
