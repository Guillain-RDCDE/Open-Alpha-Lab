"""The synthetic panel is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downside_beta import data  # noqa: E402


def test_synthetic_shape_and_finiteness(null):
    rdf, mkt, truth = null
    assert rdf.shape == (truth["n_days"], truth["n_firms"])
    assert len(mkt) == truth["n_days"]
    assert np.isfinite(rdf.to_numpy()).all()
    assert np.isfinite(mkt.to_numpy()).all()
    # index aligns between panel and market
    assert rdf.index.equals(mkt.index)


def test_synthetic_is_deterministic():
    a, ma, _ = data.synthetic_panel(n_days=300, n_firms=50, seed=5)
    b, mb, _ = data.synthetic_panel(n_days=300, n_firms=50, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ma.to_numpy(), mb.to_numpy())
    c, _, _ = data.synthetic_panel(n_days=300, n_firms=50, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_gamma_is_nonnegative_and_planted():
    """The extra downside loading is non-negative (firms only crash *extra* on down days)."""
    _, _, truth = data.synthetic_panel(seed=332)
    assert (truth["gamma"] >= 0).all()
    assert truth["gamma"].std() > 0  # there is genuine cross-sectional dispersion


def test_down_day_drag_is_neutralised_in_null():
    """In the null, alpha exactly offsets the mechanical down-day drag, so high-gamma
    firms do NOT have a systematically lower mean return."""
    rdf, mkt, truth = data.synthetic_panel(downside_premium=0.0, seed=332)
    g = truth["gamma"]
    mean_ret = rdf.mean(axis=0).to_numpy()
    # correlation between gamma and average return should be ~0 in the null
    c = np.corrcoef(g, mean_ret)[0, 1]
    assert abs(c) < 0.25


def test_priced_premium_raises_high_gamma_returns():
    """With a positive premium, higher-gamma firms earn more on average."""
    rdf, mkt, truth = data.synthetic_panel(downside_premium=0.0025, seed=332)
    g = truth["gamma"]
    mean_ret = rdf.mean(axis=0).to_numpy()
    c = np.corrcoef(g, mean_ret)[0, 1]
    assert c > 0.2


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(
            fetch=False,
            panel_path=str(tmp_path / "nope.parquet"),
            market_path=str(tmp_path / "nope_mkt.parquet"),
        )


def test_fingerprint_stable_and_content_sensitive(null):
    rdf, mkt, _ = null
    assert data.fingerprint(rdf) == data.fingerprint(rdf)
    other, _, _ = data.synthetic_panel(downside_premium=0.0, seed=99)
    assert data.fingerprint(rdf) != data.fingerprint(other)


# ---- real cache, gated (skips offline) ------------------------------------
@pytest.mark.skipif(
    not (os.path.exists(data.PANEL_CACHE) and os.path.exists(data.MARKET_CACHE)),
    reason="offline CI: no real cache present",
)
def test_load_real_from_cache_shape():
    rdf, mkt = data.load_real(fetch=False)
    assert rdf.shape[0] > 100 and rdf.shape[1] >= 5
    assert rdf.index.equals(mkt.index) or mkt.index.isin(rdf.index).all()
