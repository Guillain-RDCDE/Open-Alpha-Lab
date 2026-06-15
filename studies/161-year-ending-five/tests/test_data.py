"""The synthetic tape is well-formed and deterministic; the Shiller loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from year_ending_five import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"]
    assert set(df.columns) >= {"log_ret", "ret_pct", "last_digit"}
    assert df.index.name == "year"


def test_synthetic_last_digit_correct(null_tape):
    df, _ = null_tape
    assert (df["last_digit"] == df.index % 10).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=100, digit5_boost=0.0, seed=10)
    b, _ = data.synthetic_annual(n_years=100, digit5_boost=0.0, seed=10)
    assert np.allclose(a["ret_pct"].to_numpy(), b["ret_pct"].to_numpy())
    c, _ = data.synthetic_annual(n_years=100, digit5_boost=0.0, seed=11)
    assert not np.allclose(a["ret_pct"].to_numpy(), c["ret_pct"].to_numpy())


def test_boost_plants_higher_digit5_mean(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    mean5_null = df_null[df_null["last_digit"] == 5]["ret_pct"].mean()
    mean5_boost = df_boost[df_boost["last_digit"] == 5]["ret_pct"].mean()
    # The boosted tape should have a meaningfully higher digit-5 mean.
    assert mean5_boost > mean5_null + 5.0  # at least 5 pp higher


def test_boost_does_not_affect_other_digits(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    # Non-5 digits should have the same means (same seed, same noise for those years).
    for d in range(10):
        if d == 5:
            continue
        m_null = df_null[df_null["last_digit"] == d]["ret_pct"].mean()
        m_boost = df_boost[df_boost["last_digit"] == d]["ret_pct"].mean()
        # Should be identical (same rng state for non-5 years, same noise draws).
        assert abs(m_null - m_boost) < 1e-8


def test_each_digit_roughly_represented(null_tape):
    df, truth = null_tape
    counts = df.groupby("last_digit").size()
    # With 150 years starting at 1876, each digit should appear ~15 times.
    assert counts.min() >= 14
    assert counts.max() <= 16


def test_fingerprint_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    df2, _ = data.synthetic_annual(n_years=150, digit5_boost=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(df2)


def test_shiller_loader_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_shiller(cache_path=str(tmp_path / "no_such_file.parquet"))
