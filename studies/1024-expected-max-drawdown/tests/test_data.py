"""Data-layer tests for Study 1024 — synthetic determinism offline, bundled tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promised import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover the logic")


def _acf_sq(r, lag=1):
    x = np.asarray(r) ** 2
    x = x - x.mean()
    return float((x[lag:] * x[:-lag]).sum() / (x * x).sum())


def test_synthetic_is_deterministic_and_seed_sensitive():
    a, _ = data.synthetic_returns(n_years=5, seed=1)
    b, _ = data.synthetic_returns(n_years=5, seed=1)
    c, _ = data.synthetic_returns(n_years=5, seed=2)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_shape_index_and_horizon():
    r, truth = data.synthetic_returns(n_years=8, seed=3)
    assert len(r) == 8 * data.TRADING_DAYS_PER_YEAR == truth["n"]
    assert isinstance(r.index, pd.DatetimeIndex) and r.index.is_monotonic_increasing
    assert r.index[-1] < pd.Timestamp("2262-01-01")
    m, _ = data.synthetic_returns(n_years=10, periods_per_year=12, seed=3)
    assert len(m) == 120 and (m.index.is_month_end).all()


def test_null_world_is_iid_gaussian():
    r, truth = data.synthetic_returns(n_years=60, signal_strength=0.0, seed=4)
    assert truth["alpha"] == 0.0 and truth["beta"] == 0.0 and np.isinf(truth["nu"])
    assert abs(pd.Series(r).kurt()) < 0.15
    assert abs(_acf_sq(r)) < 0.03
    assert r.std() == pytest.approx(truth["sigma"], rel=0.03)


def test_planted_world_clusters_and_has_fat_tails_at_the_same_variance():
    r1, t1 = data.synthetic_returns(n_years=60, signal_strength=1.0, seed=4)
    assert pd.Series(r1).kurt() > 2.0
    assert _acf_sq(r1) > 0.05
    assert t1["persistence"] == pytest.approx(0.997)
    # the moments a Gaussian model "knows" are the same in both worlds
    _, t0 = data.synthetic_returns(n_years=60, signal_strength=0.0, seed=4)
    assert t1["sigma"] == t0["sigma"] and t1["mu"] == t0["mu"]


def test_signal_strength_scales_the_departures():
    _, th = data.synthetic_returns(n_years=1, signal_strength=0.5, seed=1)
    _, t1 = data.synthetic_returns(n_years=1, signal_strength=1.0, seed=1)
    assert 0 < th["alpha"] < t1["alpha"]
    assert th["nu"] > t1["nu"]


def test_declared_tapes_and_as_of():
    assert set(data.AS_OF_TAPE) == set(data.PERIODS_PER_YEAR) == set(data.LABELS)
    assert data.AS_OF == max(data.AS_OF_TAPE.values())
    for t in data.AS_OF_TAPE:
        assert len(data.sha_pin(t)) == 12
    assert "SURVIVOR" in data.LABELS["stocks"]
    assert "price index" in data.LABELS["sp500"] and "total return" in data.LABELS["ff_market"]


def test_have_real_is_a_bool():
    assert isinstance(data.have_real(), bool)


@needs_real
def test_real_tapes_load_cut_at_as_of_with_no_partial_month():
    for t in ("sp500", "nasdaq", "ff_market"):
        r = data.load_tape(t)
        assert r.index[-1] <= pd.Timestamp(data.AS_OF_TAPE[t])
        assert r.index.is_monotonic_increasing and not r.index.has_duplicates
        assert np.isfinite(r).all()
    sp = data.load_sp500()
    assert sp.index[-1] == pd.Timestamp("2022-11-30")
    ff = data.load_ff_market()
    assert ff.index[0] == pd.Timestamp("1926-07-31") and len(ff) == 1109
    S = data.load_stocks()
    assert S.shape[1] == 20


@needs_real
def test_real_fingerprints_are_stable():
    a, b = data.load_sp500(), data.load_sp500()
    assert data.fingerprint(a) == data.fingerprint(b)
