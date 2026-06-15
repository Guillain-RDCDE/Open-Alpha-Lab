"""The synthetic panel is well-formed and deterministic; real loaders are cache-safe."""

import os
import sys
import string

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphabetical_bias import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic panel shape and contents
# ---------------------------------------------------------------------------
def test_synthetic_panel_columns(null_panel):
    panel, truth = null_panel
    expected = {"ticker", "date", "letter", "is_early", "ret", "volume"}
    assert expected.issubset(set(panel.columns))


def test_synthetic_panel_row_count(null_panel):
    panel, truth = null_panel
    expected_rows = truth["n_stocks"] * truth["n_months"]
    assert len(panel) == expected_rows


def test_synthetic_panel_is_deterministic():
    a, _ = data.synthetic_cross(n_stocks=50, n_months=12, seed=7)
    b, _ = data.synthetic_cross(n_stocks=50, n_months=12, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_cross(n_stocks=50, n_months=12, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_synthetic_letters_are_valid(null_panel):
    panel, _ = null_panel
    valid = set(string.ascii_uppercase)
    assert set(panel["letter"].unique()).issubset(valid)


def test_early_flag_consistent(null_panel):
    panel, _ = null_panel
    for _, row in panel.sample(min(100, len(panel)), random_state=0).iterrows():
        expected_early = row["letter"] in data.EARLY_LETTERS
        assert row["is_early"] == expected_early


def test_return_bias_increases_early_returns():
    """Planting a bias actually shifts early-alphabet returns upward."""
    null, _ = data.synthetic_cross(n_stocks=200, n_months=120, return_bias_bps=0.0, seed=42)
    biased, _ = data.synthetic_cross(n_stocks=200, n_months=120, return_bias_bps=50.0, seed=42)
    null_mean = null[null["is_early"]]["ret"].mean()
    biased_mean = biased[biased["is_early"]]["ret"].mean()
    assert biased_mean > null_mean + 10e-4  # at least 10 bps more


def test_volume_bias_increases_early_volume():
    """Planting a volume bias actually shifts early-alphabet volume upward."""
    null, _ = data.synthetic_cross(n_stocks=200, n_months=60, volume_bias=0.0, seed=42)
    biased, _ = data.synthetic_cross(n_stocks=200, n_months=60, volume_bias=0.50, seed=42)
    null_vol = null[null["is_early"]]["volume"].mean()
    biased_vol = biased[biased["is_early"]]["volume"].mean()
    assert biased_vol > null_vol * 1.2  # at least 20% more


def test_volume_positive(null_panel):
    panel, _ = null_panel
    assert (panel["volume"] > 0).all()


def test_fingerprint_stable_and_content_sensitive(null_panel):
    panel, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_cross(n_stocks=150, n_months=60, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))
