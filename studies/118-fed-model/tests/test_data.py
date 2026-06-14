"""The synthetic panel is well-formed and deterministic; the Shiller loader validates cleanly."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_model import data  # noqa: E402


def test_synthetic_shape_and_columns(null_panel):
    panel, truth = null_panel
    assert len(panel) == truth["n_months"]
    expected_cols = {"EP", "rate", "fed_spread", "infl", "real_rate", "ep_vs_real", "fwd1", "fwd10"}
    assert expected_cols.issubset(set(panel.columns))


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_months=100, predicts=0.5, seed=42)
    b, _ = data.synthetic_panel(n_months=100, predicts=0.5, seed=42)
    assert np.allclose(a["EP"].to_numpy(), b["EP"].to_numpy())
    c, _ = data.synthetic_panel(n_months=100, predicts=0.5, seed=43)
    assert not np.allclose(a["EP"].to_numpy(), c["EP"].to_numpy())


def test_synthetic_ep_and_rate_are_positive(null_panel):
    panel, _ = null_panel
    assert (panel["EP"] > 0).all()
    assert (panel["rate"] > 0).all()


def test_synthetic_spread_matches_ep_minus_rate(signal_panel):
    panel, _ = signal_panel
    diff = (panel["fed_spread"] - (panel["EP"] - panel["rate"])).abs()
    assert diff.max() < 1e-12


def test_synthetic_null_has_near_zero_slope(null_panel):
    """When predicts=0 the OLS slope should be near zero (no planted signal)."""
    from fed_model import strategy as st
    panel, _ = null_panel
    sub = panel.dropna(subset=["fed_spread", "fwd10"])
    result = st.ols_hac(sub["fed_spread"].values, sub["fwd10"].values)
    assert abs(result["slope"]) < 0.5  # loose: noise could push it a bit, but not large


def test_synthetic_signal_has_positive_slope(signal_panel):
    """When predicts=1.5 the OLS slope on 10yr returns should be clearly positive."""
    from fed_model import strategy as st
    panel, _ = signal_panel
    sub = panel.dropna(subset=["fed_spread", "fwd10"])
    result = st.ols_hac(sub["fed_spread"].values, sub["fwd10"].values)
    assert result["slope"] > 0.5


def test_shiller_load_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_shiller(cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    panel, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_panel(n_months=600, predicts=0.0, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)
