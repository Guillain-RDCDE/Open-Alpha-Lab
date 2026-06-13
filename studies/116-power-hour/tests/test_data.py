"""The synthetic panel is well-formed, deterministic, and the continuation knob works;
the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_hour import data  # noqa: E402


def test_synthetic_shape_and_columns(null_panel):
    panel, truth = null_panel
    assert len(panel) == truth["n_sessions"]
    assert list(panel.columns) == data.FEATURE_COLUMNS
    assert (panel["morning_ret"].notna()).all()
    assert (panel["last_ret"].notna()).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_sessions(n_sessions=100, continuation=0.1, seed=5)
    b, _ = data.synthetic_sessions(n_sessions=100, continuation=0.1, seed=5)
    assert np.allclose(a["last_ret"].to_numpy(), b["last_ret"].to_numpy())
    c, _ = data.synthetic_sessions(n_sessions=100, continuation=0.1, seed=6)
    assert not np.allclose(a["last_ret"].to_numpy(), c["last_ret"].to_numpy())


def test_null_has_near_zero_correlation(null_panel):
    """With continuation=0 the morning and last-hour return should be nearly uncorrelated."""
    panel, _ = null_panel
    r = np.corrcoef(panel["morning_ret"], panel["last_ret"])[0, 1]
    assert abs(r) < 0.12


def test_continuation_induces_positive_correlation(trending_panel):
    """With continuation=0.30 the morning should positively predict the last hour."""
    panel, _ = trending_panel
    r = np.corrcoef(panel["morning_ret"], panel["last_ret"])[0, 1]
    assert r > 0.1


def test_reversal_induces_negative_correlation(reversal_panel):
    """With continuation=-0.30 the morning should negatively predict the last hour."""
    panel, _ = reversal_panel
    r = np.corrcoef(panel["morning_ret"], panel["last_ret"])[0, 1]
    assert r < -0.1


def test_full_ret_is_chained_product(null_panel):
    """full_ret should equal the simple-return product of morning and last-hour moves."""
    panel, _ = null_panel
    expected = (1 + panel["morning_ret"]) * (1 + panel["last_ret"]) - 1
    assert np.allclose(panel["full_ret"].to_numpy(), expected.to_numpy(), atol=1e-10)


def test_fetch_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily_panel("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    panel, _ = null_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_sessions(n_sessions=500, continuation=0.0, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)
