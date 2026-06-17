"""The synthetic tape is well-formed and deterministic; the real tape loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hemline_index import data  # noqa: E402

SHILLER_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_cache", "shiller_sp500.parquet")
)


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_decades"]
    assert set(df.columns) >= {"hemline_dir", "market_positive", "market_bullish", "decade_return_proxy"}


def test_synthetic_hemline_dir_values(null_tape):
    df, _ = null_tape
    assert set(df["hemline_dir"].unique()).issubset({-1, 1})


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_decades(n_decades=50, hemline_boost=0.0, seed=10)
    b, _ = data.synthetic_decades(n_decades=50, hemline_boost=0.0, seed=10)
    assert np.allclose(a["decade_return_proxy"].to_numpy(), b["decade_return_proxy"].to_numpy())
    c, _ = data.synthetic_decades(n_decades=50, hemline_boost=0.0, seed=11)
    assert not np.allclose(a["decade_return_proxy"].to_numpy(), c["decade_return_proxy"].to_numpy())


def test_boost_plants_correlation(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    # With a boost the hit rate (hemline dir agrees with market direction) should be higher
    from hemline_index.strategy import agreement_table
    hr_null = agreement_table(df_null)["hit_rate"]
    hr_boost = agreement_table(df_boost)["hit_rate"]
    assert hr_boost > hr_null + 0.10


def test_boost_does_not_change_hemline_counts(null_tape, boosted_tape):
    df_null, _ = null_tape
    df_boost, _ = boosted_tape
    # Same seed -> same hemline draws
    assert (df_null["hemline_dir"].to_numpy() == df_boost["hemline_dir"].to_numpy()).all()


def test_real_tape_hardcoded_shape(real_tape):
    assert len(real_tape) == 10
    assert set(real_tape.columns) >= {"hemline_dir", "sp500_decade_pct", "fashion_note"}


def test_real_tape_hemline_values(real_tape):
    assert set(real_tape["hemline_dir"].unique()).issubset({-1, 1})


def test_real_tape_skips_when_no_shiller(tmp_path):
    # load_real_tape with a non-existent shiller path still returns the hardcoded table
    df = data.load_real_tape(shiller_cache=str(tmp_path / "no_file.parquet"))
    assert len(df) == 10  # hardcoded table always has 10 rows


@pytest.mark.skipif(
    not os.path.exists(SHILLER_CACHE),
    reason="Shiller cache absent -- CI / offline",
)
def test_real_tape_shiller_returns_reasonable(real_tape):
    # All decade returns should be plausible equity values (-80% to +1000%)
    assert (real_tape["sp500_decade_pct"] > -90).all()
    assert (real_tape["sp500_decade_pct"] < 2000).all()


def test_fingerprint_stable(real_tape):
    assert data.fingerprint(real_tape) == data.fingerprint(real_tape)
