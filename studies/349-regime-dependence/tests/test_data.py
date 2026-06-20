"""The synthetic regime panel is well-formed and deterministic; the real loader is
cache- and offline-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_dependence import data  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
PRICE_CACHE = os.path.join(CACHE, "regime_prices.parquet")


# ---- synthetic shape & determinism (never skip) ----------------------------
def test_synthetic_shape(planted_panel):
    df, regime, truth = planted_panel
    assert df.shape == (truth["n"], 2)
    assert list(df.columns) == ["durable", "regime_fit"]
    assert len(regime) == truth["n"]
    assert np.isfinite(df.to_numpy()).all()


def test_regime_labels_cover_all_regimes(planted_panel):
    df, regime, truth = planted_panel
    assert sorted(regime.unique()) == list(range(truth["n_regimes"]))
    # each regime has the planted length
    counts = regime.value_counts()
    assert (counts == truth["months_per_regime"]).all()


def test_synthetic_is_deterministic():
    a, ra, _ = data.synthetic_regimes(seed=7)
    b, rb, _ = data.synthetic_regimes(seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.array_equal(ra.to_numpy(), rb.to_numpy())
    c, _, _ = data.synthetic_regimes(seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_hot_regime_has_higher_regime_fit_mean(planted_panel):
    """The regime-fitted strategy earns its large mean in exactly the planted hot regime."""
    df, regime, truth = planted_panel
    hot = truth["hot_regime"]
    means = df["regime_fit"].groupby(regime.to_numpy()).mean()
    assert means.idxmax() == hot
    # and the hot mean is far above the median of the others
    others = means.drop(hot)
    assert means[hot] > others.max() + 0.005


def test_durable_mean_is_flat_across_regimes(planted_panel):
    """The durable strategy's mean is roughly constant across regimes (a stable edge)."""
    df, regime, _ = planted_panel
    means = df["durable"].groupby(regime.to_numpy()).mean()
    assert (means.max() - means.min()) < 0.010  # no single regime dominates


def test_index_uses_period_range_not_overflow():
    """The decorative monthly index is built without overflowing ns-Timestamps."""
    df, _, _ = data.synthetic_regimes(n_regimes=3, months_per_regime=60)
    assert df.index.is_monotonic_increasing
    assert df.index[0].year >= 1900


def test_fingerprint_stable_and_sensitive(planted_panel):
    df, _, _ = planted_panel
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _, _ = data.synthetic_regimes(seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)


# ---- real loader: cache contract -------------------------------------------
def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(not os.path.exists(PRICE_CACHE), reason="offline CI: no real cache")
def test_real_series_loads_from_cache():
    rets = data.load_real(fetch=False)
    assert list(rets.columns) == ["equity", "bond"]
    assert np.isfinite(rets.to_numpy()).all()
