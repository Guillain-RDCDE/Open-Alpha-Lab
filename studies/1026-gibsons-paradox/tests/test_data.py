"""Data-layer tests for Study 1026 — synthetic determinism offline, real tapes gated."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gibson import data  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="arch / statsmodels tapes not installed")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_world(n_months=240, seed=1026)
    b, _ = data.synthetic_world(n_months=240, seed=1026)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_seed_sensitive():
    a, _ = data.synthetic_world(n_months=240, seed=1026)
    b, _ = data.synthetic_world(n_months=240, seed=1027)
    assert not np.allclose(a["aaa"].to_numpy(), b["aaa"].to_numpy())


def test_synthetic_shape_index_and_horizon():
    df, truth = data.synthetic_world(n_months=300, seed=1026)
    assert len(df) == 300 == truth["n_months"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing
    assert df.index[-1] < pd.Timestamp("2262-01-01")
    for c in ("aaa", "logp", "pi", "gap_exp", "gap_rol", "rf"):
        assert c in df.columns
    assert (df["rf"] >= 0).all()


def test_price_path_is_identical_across_worlds():
    """The two worlds differ only in the yield law — the price path is shared."""
    f, _ = data.synthetic_world(n_months=240, signal_strength=0.0, seed=1026)
    g, _ = data.synthetic_world(n_months=240, signal_strength=1.0, seed=1026)
    np.testing.assert_allclose(f["logp"], g["logp"])
    np.testing.assert_allclose(f["gap_exp"].dropna(), g["gap_exp"].dropna())
    assert not np.allclose(f["aaa"], g["aaa"])


def test_signal_strength_is_a_blend():
    f, _ = data.synthetic_world(n_months=240, signal_strength=0.0, seed=1026)
    g, _ = data.synthetic_world(n_months=240, signal_strength=1.0, seed=1026)
    m, t = data.synthetic_world(n_months=240, signal_strength=0.5, seed=1026)
    np.testing.assert_allclose(m["aaa"], 0.5 * f["aaa"] + 0.5 * g["aaa"])
    assert t["gibson_weight"] == 0.5 and t["fisher_weight"] == 0.5


def test_price_gap_is_past_only():
    """Changing the future must not change the gap at t."""
    rng = np.random.default_rng(0)
    lp = pd.Series(np.cumsum(rng.normal(0.3, 0.2, 300)))
    g1 = data.price_gap(lp, "expanding", min_obs=60)
    lp2 = lp.copy()
    lp2.iloc[200:] += 50.0
    g2 = data.price_gap(lp2, "expanding", min_obs=60)
    np.testing.assert_allclose(g1.iloc[:200].dropna(), g2.iloc[:200].dropna())
    r1 = data.price_gap(lp, "rolling", window=120)
    r2 = data.price_gap(lp2, "rolling", window=120)
    np.testing.assert_allclose(r1.iloc[:200].dropna(), r2.iloc[:200].dropna())


def test_price_gap_of_a_pure_trend_is_zero():
    lp = pd.Series(5.0 + 0.25 * np.arange(200.0))
    g = data.price_gap(lp, "expanding", min_obs=20).dropna()
    assert np.abs(g).max() < 1e-8
    r = data.price_gap(lp, "rolling", window=40).dropna()
    assert np.abs(r).max() < 1e-8


def test_price_gap_rejects_unknown_method():
    with pytest.raises(ValueError):
        data.price_gap(pd.Series(np.arange(10.0)), "magic")


def test_constants_are_sane():
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.US_START)
    assert pd.Timestamp(data.GREAT_INFLATION[1]) < pd.Timestamp(data.DISINFLATION[0])
    assert pd.Timestamp(data.AS_OF) < pd.Timestamp("2026-10-01")


@needs_real
def test_us_monthly_tape_is_pinned_and_complete():
    us = data.load_us_monthly()
    assert us.index[0] == pd.Timestamp("1957-01-31")
    assert us.index[-1] == pd.Timestamp(data.AS_OF)
    assert us.index.is_monotonic_increasing and not us.index.has_duplicates
    assert us[["aaa", "cpi", "rf"]].notna().all().all()
    assert 2.0 < us["aaa"].min() and us["aaa"].max() < 16.0
    assert data.fingerprint(us[["aaa", "baa", "cpi", "rf"]]) == "d269fafaad79"


@needs_real
def test_quarterly_and_german_tapes_load():
    uq = data.load_us_quarterly()
    de = data.load_germany()
    assert uq.index[0].year == 1959 and uq.index[-1].year == 2009
    assert de.index[0].year == 1972 and de.index[-1].year == 1998
    # Dp is seasonal (negative lag-1 autocorrelation); the 4q inflation measure is smooth
    assert de["Dp"].autocorr(1) < 0.0
    assert de["pi"].dropna().autocorr(1) > 0.7


@needs_real
def test_no_price_index_before_1957():
    """The study's framing limit, asserted: the yield tape is older than any price index."""
    y = data.load_yields()
    us = data.load_us_monthly()
    assert y.index[0].year == 1919
    assert us["cpi"].first_valid_index().year == 1957


@needs_real
def test_provenance_lists_sha_pins():
    rows = data.provenance()
    pinned = [r for r in rows if r["package"] == "arch"]
    assert len(pinned) == 3 and all(len(r["sha256"]) == 64 for r in pinned)
