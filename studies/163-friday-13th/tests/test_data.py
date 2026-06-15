"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from friday_13th import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert "ret" in df.columns
    assert "is_f13" in df.columns
    assert "is_f6" in df.columns
    assert "is_friday" in df.columns
    assert len(df) == truth["n_days"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=30, f13_effect=0.0, seed=42)
    b, _ = data.synthetic_daily(n_years=30, f13_effect=0.0, seed=42)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_daily(n_years=30, f13_effect=0.0, seed=99)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_friday_13th_count_is_positive(null_tape):
    df, truth = null_tape
    n_f13 = int(df["is_f13"].sum())
    assert n_f13 > 0
    # With 80 years of business days, expect ~136 Friday-13ths.
    assert 50 < n_f13 < 200
    assert n_f13 == truth["n_f13"]


def test_f13_implies_friday(null_tape):
    """Every Friday-13th day must also be a Friday (weekday == 4)."""
    df, _ = null_tape
    f13_idx = df.index[df["is_f13"]]
    assert (f13_idx.weekday == 4).all()
    assert (f13_idx.day == 13).all()


def test_f6_implies_friday_6th(null_tape):
    df, _ = null_tape
    f6_idx = df.index[df["is_f6"]]
    assert (f6_idx.weekday == 4).all()
    assert (f6_idx.day == 6).all()


def test_planted_effect_shifts_mean(null_tape, negative_tape):
    """The f13_effect knob really shifts the mean return on Friday-13th days."""
    df_null, _ = null_tape
    df_neg, _ = negative_tape
    mean_null = df_null.loc[df_null["is_f13"], "ret"].mean()
    mean_neg = df_neg.loc[df_neg["is_f13"], "ret"].mean()
    # Planted negative effect should lower the mean.
    assert mean_neg < mean_null - 0.001


def test_fetch_gspc_cache_only_raises_without_cache(tmp_path, monkeypatch):
    """Without any cache (shared or local), fetch_gspc must raise FileNotFoundError."""
    # Temporarily redirect the shared cache path to a non-existent location.
    monkeypatch.setattr(data, "_GSPC_SHARED", str(tmp_path / "no_such_file.parquet"))
    monkeypatch.setattr(data, "_GSPC_LOCAL", str(tmp_path / "no_such_local.parquet"))
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_years=80, f13_effect=0.0, seed=999)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_is_friday_13th_helper():
    """Pure date-arithmetic helper flags the right days."""
    idx = data.pd.date_range("2023-01-01", "2024-01-01")
    mask = data.is_friday_13th(idx)
    # 2023-01-13 is a Friday.
    assert mask[data.pd.Timestamp("2023-01-13")]
    # 2023-01-06 is a Friday but NOT the 13th.
    assert not mask[data.pd.Timestamp("2023-01-06")]
    # 2023-09-13 is a Wednesday — not a Friday.
    assert not mask[data.pd.Timestamp("2023-09-13")]
