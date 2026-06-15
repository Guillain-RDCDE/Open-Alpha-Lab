"""The synthetic tape is well-formed, deterministic, and the effect knob works."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megacap_concentration import data  # noqa: E402


def test_synthetic_shape_and_columns(null_panel):
    df, truth = null_panel
    assert set(["year", "ticker", "rank", "ret_gross", "is_top_n"]).issubset(df.columns)
    assert len(df) == truth["n_years"] * truth["n_stocks"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=10, n_stocks=50, top_n=5, effect=0.0, seed=42)
    b, _ = data.synthetic_annual(n_years=10, n_stocks=50, top_n=5, effect=0.0, seed=42)
    assert np.allclose(a["ret_gross"].to_numpy(), b["ret_gross"].to_numpy())
    c, _ = data.synthetic_annual(n_years=10, n_stocks=50, top_n=5, effect=0.0, seed=43)
    assert not np.allclose(a["ret_gross"].to_numpy(), c["ret_gross"].to_numpy())


def test_synthetic_rank_is_valid_permutation(null_panel):
    df, truth = null_panel
    for yr, grp in df.groupby("year"):
        ranks = sorted(grp["rank"].tolist())
        assert ranks == list(range(1, truth["n_stocks"] + 1)), \
            f"Year {yr}: ranks should be a permutation of 1..{truth['n_stocks']}"


def test_synthetic_top_n_count_per_year(null_panel):
    df, truth = null_panel
    for yr, grp in df.groupby("year"):
        n_topn = grp["is_top_n"].sum()
        assert n_topn == truth["top_n"], \
            f"Year {yr}: expected {truth['top_n']} top-N stocks, got {n_topn}"


def test_synthetic_effect_knob_shifts_returns():
    """With a large planted effect, top-N should clearly outperform the rest."""
    _, truth_null = data.synthetic_annual(
        n_years=50, n_stocks=200, top_n=10, effect=0.0, seed=177
    )
    df_big, _ = data.synthetic_annual(
        n_years=50, n_stocks=200, top_n=10, effect=0.20, seed=177
    )
    topn_mean = df_big[df_big["is_top_n"]]["ret_gross"].mean()
    rest_mean = df_big[~df_big["is_top_n"]]["ret_gross"].mean()
    # With effect=0.20 over 50 years the top-N must handily beat the rest
    assert topn_mean > rest_mean + 0.05, \
        f"Expected top-N to outperform rest by >5ppt; got {topn_mean:.4f} vs {rest_mean:.4f}"


def test_load_real_raises_without_cache(tmp_path):
    """load_real in cache-only mode raises FileNotFoundError when the cache is absent."""
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    df, _ = null_panel
    import pandas as pd
    # Rename to match fingerprint API expectation
    df2 = df.rename(columns={"ret_gross": "ret_annual"})
    fp1 = data.fingerprint(df2)
    fp2 = data.fingerprint(df2)
    assert fp1 == fp2
    # A different tape should have a different fingerprint
    df_other, _ = data.synthetic_annual(n_years=18, n_stocks=100, top_n=10, effect=0.0, seed=999)
    df_other2 = df_other.rename(columns={"ret_gross": "ret_annual"})
    assert data.fingerprint(df2) != data.fingerprint(df_other2)
