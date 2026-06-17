"""Tests for the data layer of Study 281 (El-Nino).

All tests are offline and deterministic -- hardcoded ONI table and synthetic
generator only; no network, no price cache.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from el_nino import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded ONI table
# ---------------------------------------------------------------------------
def test_oni_columns(oni):
    expected = {"enso_year", "oni_ndj", "phase", "warm", "cold"}
    assert expected.issubset(set(oni.columns))


def test_oni_years_cover_range(oni):
    assert oni["enso_year"].min() == 1950
    assert oni["enso_year"].max() == 2024
    # contiguous, no gaps
    assert list(oni["enso_year"]) == list(range(1950, 2025))


def test_oni_phases_valid(oni):
    assert set(oni["phase"].unique()).issubset({"El Nino", "La Nina", "Neutral"})


def test_oni_threshold_classification(oni):
    """Phase must respect the +/-0.5 ONI threshold exactly."""
    for _, row in oni.iterrows():
        v = row["oni_ndj"]
        if v >= 0.5:
            assert row["phase"] == "El Nino"
        elif v <= -0.5:
            assert row["phase"] == "La Nina"
        else:
            assert row["phase"] == "Neutral"


def test_oni_known_events(oni):
    """Spot-check famous ENSO winters."""
    # 1997-98 super El Nino
    assert oni.loc[oni["enso_year"] == 1997, "phase"].iloc[0] == "El Nino"
    assert oni.loc[oni["enso_year"] == 1997, "oni_ndj"].iloc[0] >= 2.0
    # 1988-89 strong La Nina
    assert oni.loc[oni["enso_year"] == 1988, "phase"].iloc[0] == "La Nina"
    # 2015-16 super El Nino
    assert oni.loc[oni["enso_year"] == 2015, "phase"].iloc[0] == "El Nino"


def test_oni_phase_mix_reasonable(oni):
    """Each phase should appear a sensible number of times."""
    counts = oni["phase"].value_counts()
    for ph in ("El Nino", "La Nina", "Neutral"):
        assert counts[ph] >= 10
    assert counts.sum() == len(oni)


def test_oni_warm_cold_flags_consistent(oni):
    assert (oni["warm"] == (oni["phase"] == "El Nino")).all()
    assert (oni["cold"] == (oni["phase"] == "La Nina")).all()


def test_oni_table_is_copy():
    t1 = data.oni_table()
    t2 = data.oni_table()
    t1.loc[0, "phase"] = "XXX"
    assert t2.loc[0, "phase"] != "XXX"


def test_oni_threshold_param_changes_labels():
    """A laxer threshold reclassifies some neutral years."""
    strict = data.oni_table(threshold=0.5)
    lax = data.oni_table(threshold=0.3)
    n_neutral_strict = (strict["phase"] == "Neutral").sum()
    n_neutral_lax = (lax["phase"] == "Neutral").sum()
    assert n_neutral_lax < n_neutral_strict


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    df, truth = synthetic_null
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"year", "asset_return", "oni_ndj", "phase", "warm", "cold"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=42)
    b, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=1)
    b, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=2)
    assert not np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())


def test_synthetic_null_has_no_signal():
    """On the null tape, El Nino / La Nina mean returns should be close."""
    df, _ = data.synthetic_panel(n_years=2000, premium_bps=0.0, seed=281)
    m_el = df.loc[df["phase"] == "El Nino", "ret"].mean()
    m_la = df.loc[df["phase"] == "La Nina", "ret"].mean()
    assert abs(m_el - m_la) < 0.02


def test_synthetic_signal_creates_gap():
    """A large planted premium creates a measurable El Nino > La Nina gap."""
    df, _ = data.synthetic_panel(n_years=600, premium_bps=2_000.0, seed=281)
    m_el = df.loc[df["phase"] == "El Nino", "ret"].mean()
    m_la = df.loc[df["phase"] == "La Nina", "ret"].mean()
    assert m_el > m_la + 0.10


def test_fingerprint_stable_and_content_sensitive():
    df1, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=10)
    df2, _ = data.synthetic_panel(n_years=40, premium_bps=0.0, seed=11)
    df1 = df1.rename(columns={"ret": "asset_return"})
    df2 = df2.rename(columns={"ret": "asset_return"})
    assert data.fingerprint(df1) == data.fingerprint(df1)
    assert data.fingerprint(df1) != data.fingerprint(df2)


# ---------------------------------------------------------------------------
# Real loader -- offline behaviour
# ---------------------------------------------------------------------------
def test_fetch_real_raises_without_cache(tmp_path):
    """With no cache and fetch=False, fetch_real raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_real(asset="equity", cache_dir=str(tmp_path), fetch=False)


def test_fetch_real_bad_asset():
    with pytest.raises(ValueError):
        data.fetch_real(asset="banana")
