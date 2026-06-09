"""The synthetic tape bakes in exactly what the study needs: a persistent vol regime (so variance
is forecastable and storms carry risk-without-return), plus a flat-vol null where it doesn't."""

import numpy as np

from storm_shy import data


def test_deterministic_given_seed():
    a, _ = data.synthetic_prices(seed=16)
    b, _ = data.synthetic_prices(seed=16)
    assert (a.to_numpy() == b.to_numpy()).all()


def test_prices_are_positive_and_dated(regime):
    close, truth = regime
    assert (close > 0).all()
    assert close.index.name == "date"
    assert len(close) == truth.n_bars


def test_regime_clusters_volatility(regime_returns, flat_returns):
    """A persistent regime makes |returns| autocorrelated (clustering); flat vol does not."""
    def abs_autocorr(r):
        a = (r.abs() - r.abs().mean()).to_numpy()
        return float(a[:-1] @ a[1:] / (a @ a))
    assert abs_autocorr(regime_returns) > 0.10        # clear clustering
    assert abs(abs_autocorr(flat_returns)) < 0.06     # essentially none


def test_theoretical_gain_ceiling():
    """The perfect-foresight Sharpe multiple is > 1 with a regime, exactly 1 without."""
    _, reg = data.synthetic_prices(sigma_lo=0.006, sigma_hi=0.024, seed=16)
    _, flt = data.synthetic_prices(sigma_lo=0.013, sigma_hi=0.013, seed=16)
    assert reg.has_regime and reg.theoretical_sharpe_gain > 1.3
    assert not flt.has_regime
    assert np.isclose(flt.theoretical_sharpe_gain, 1.0)
