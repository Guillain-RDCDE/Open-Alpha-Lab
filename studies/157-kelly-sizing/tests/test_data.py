"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly_sizing import data  # noqa: E402


def test_synthetic_shape_and_type(positive_edge):
    returns, truth = positive_edge
    assert len(returns) == truth["n_days"]
    assert returns.name == "ret"
    assert isinstance(returns.index, type(returns.index))
    assert (returns.index.name == "date")


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_returns(n_years=5.0, sharpe=0.4, seed=10)
    b, _ = data.synthetic_returns(n_years=5.0, sharpe=0.4, seed=10)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_returns(n_years=5.0, sharpe=0.4, seed=11)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_zero_sharpe_has_near_zero_mean(zero_edge):
    returns, truth = zero_edge
    # With 20y of data the sample mean should be close to zero
    assert abs(returns.mean()) < 5e-4


def test_positive_sharpe_has_positive_mean(positive_edge):
    returns, truth = positive_edge
    assert returns.mean() > 0


def test_sharpe_roughly_planted(positive_edge):
    returns, truth = positive_edge
    ann_sharpe = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
    # With 20y the sample Sharpe should be in a reasonable range around 0.5.
    # The standard error of annual Sharpe over 20y is ~1/sqrt(20) ≈ 0.22,
    # so we allow +/-0.6 (about 3 standard errors of sampling error).
    assert abs(ann_sharpe - truth["sharpe"]) < 0.6


def test_fetch_spy_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_spy(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(positive_edge):
    returns, _ = positive_edge
    assert data.fingerprint(returns) == data.fingerprint(returns)
    other, _ = data.synthetic_returns(n_years=20.0, sharpe=0.5, seed=999)
    assert data.fingerprint(returns) != data.fingerprint(other)
