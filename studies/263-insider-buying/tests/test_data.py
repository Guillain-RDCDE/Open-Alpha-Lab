"""Tests for the data layer (curated ratio, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insider_buying import data  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Curated buy/sell ratio
# ---------------------------------------------------------------------------
def test_ratio_shape_and_range():
    r = data.insider_ratio()
    assert isinstance(r, pd.Series)
    assert len(r) == 276  # 2003-01 .. 2025-12 monthly
    assert r.min() >= 0.05 - 1e-9
    assert r.max() <= 1.0 + 1e-9
    assert r.notna().all()


def test_ratio_month_end_index():
    r = data.insider_ratio()
    for ts in r.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)
    assert r.index[0] == pd.Timestamp("2003-01-31")
    assert r.index[-1] == pd.Timestamp("2025-12-31")


def test_ratio_deterministic():
    a = data.insider_ratio()
    b = data.insider_ratio()
    pd.testing.assert_series_equal(a, b)


def test_ratio_spikes_present():
    """The documented buying spikes (2009-03, 2020-03) should sit near the top."""
    r = data.insider_ratio()
    assert r.loc["2009-03-31"] > 0.6, "Expected a 2009 buying spike"
    assert r.loc["2020-03-31"] > 0.6, "Expected a 2020 COVID buying spike"
    # 2021 top: insiders selling, ratio low
    assert r.loc["2021-11-30"] < 0.2, "Expected a 2021-top selling trough"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_panel(n_months=120, beta=0.05, seed=1)
    assert df.shape == (120, 2)
    assert set(df.columns) == {"bs_ratio", "fwd_ret"}
    assert truth["beta"] == 0.05


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(n_months=120, beta=0.05, seed=42)
    b, _ = data.synthetic_panel(n_months=120, beta=0.05, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_ratio_in_band():
    df, _ = data.synthetic_panel(n_months=200, beta=0.04, seed=3)
    assert df["bs_ratio"].min() >= 0.05 - 1e-9
    assert df["bs_ratio"].max() <= 1.0 + 1e-9


def test_synthetic_null_zero_beta():
    df, truth = data.synthetic_panel(n_months=120, beta=0.0, seed=10)
    assert truth["beta"] == 0.0
    assert df.shape[0] == 120


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    r = data.insider_ratio()
    fp = data.fingerprint(r)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(n_months=60, beta=0.01, seed=7)
    b, _ = data.synthetic_panel(n_months=60, beta=0.09, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard (offline by default)
# ---------------------------------------------------------------------------
def test_fetch_offline_guard(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_sp500(fetch=False, cache_path=missing)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_gspc_shape():
    g = data.fetch_sp500(fetch=False, cache_path=CACHE_PATH)
    assert g.shape[0] > 100
    assert "close" in g.columns


@requires_cache
def test_build_real_panel_no_lookahead():
    g = data.fetch_sp500(fetch=False, cache_path=CACHE_PATH)
    df = data.build_real_panel(g)
    # fwd_ret at row t equals ret at row t+1 (shift), wherever both defined.
    aligned = df["ret"].shift(-1)
    common = df["fwd_ret"].dropna().index.intersection(aligned.dropna().index)
    np.testing.assert_allclose(
        df["fwd_ret"].loc[common].to_numpy(),
        aligned.loc[common].to_numpy(),
        rtol=1e-9, atol=1e-12,
    )
