"""Data-layer tests for Study 1023 — synthetic determinism offline, real tape gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cushing import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch tapes unavailable — synthetic tests cover the logic")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_spread(seed=5)
    b, _ = data.synthetic_spread(seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_is_seed_sensitive():
    a, _ = data.synthetic_spread(seed=5)
    b, _ = data.synthetic_spread(seed=6)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_index_and_positivity():
    px, truth = data.synthetic_spread(n_months=500)
    assert list(px.columns) == ["brent", "wti"]
    assert len(px) == 500 == truth["n_months"]
    assert isinstance(px.index, pd.DatetimeIndex) and px.index.is_monotonic_increasing
    assert px.index[-1] < pd.Timestamp("2262-01-01")
    assert (px > 0).all().all()


def test_signal_strength_scales_reversion_speed():
    _, t1 = data.synthetic_spread(signal_strength=1.0)
    _, th = data.synthetic_spread(signal_strength=0.5)
    _, t0 = data.synthetic_spread(signal_strength=0.0)
    assert t0["phi"] == 1.0 and np.isinf(t0["halflife_months"])
    assert t1["phi"] < th["phi"] < 1.0
    assert t1["halflife_months"] == pytest.approx(3.0)
    assert th["halflife_months"] == pytest.approx(6.0)


def test_planted_break_moves_the_mean_where_it_says():
    px, tr = data.synthetic_spread(break_size=0.15, n_months=600, seed=3)
    x = np.log(px["brent"] / px["wti"])
    k = tr["break_index"]
    assert tr["break_date"] == px.index[k]
    assert x.iloc[k:].mean() - x.iloc[:k].mean() == pytest.approx(0.15, abs=0.03)


def test_no_break_means_no_break_metadata():
    _, tr = data.synthetic_spread(break_size=0.0)
    assert tr["break_index"] is None and tr["break_date"] is None


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_spread(seed=1)
    b, _ = data.synthetic_spread(seed=2)
    assert data.fingerprint(a) == data.fingerprint(a) and len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_declared_constants():
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.START)
    assert pd.Timestamp(data.AS_OF) == pd.Timestamp(data.AS_OF) + pd.offsets.MonthEnd(0)
    assert len(data.SHA256) == 64


@needs_real
def test_real_tape_loads_and_is_pinned():
    px = data.load_crude()
    assert list(px.columns) == ["brent", "wti"]
    assert px.index[0] == pd.Timestamp(data.START)
    assert px.index[-1] == pd.Timestamp(data.AS_OF)
    assert len(px) == 393
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates
    assert (px > 0).all().all()


@needs_real
def test_real_tape_respects_asof():
    px = data.load_crude(asof="2009-12-31")
    assert px.index[-1] == pd.Timestamp("2009-12-31")


@needs_real
def test_monthly_tape_is_a_monthly_average_of_daily_prints():
    chk = data.check_monthly_average()
    assert chk["is_monthly_average"]
    assert chk["n_months"] > 300
    assert chk["median_abs_gap_avg"] < 0.01 < chk["median_abs_gap_end"]
