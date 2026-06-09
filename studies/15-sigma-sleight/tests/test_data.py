"""The synthetic tape is what we say it is: positive prices, deterministic, and carrying a real
single-horizon mean reversion (down-moves are followed by up-tilts)."""

import numpy as np

from sigma_sleight import data


def test_prices_are_positive_and_dated(close):
    assert (close > 0).all()
    assert close.index.is_monotonic_increasing
    assert close.index.name == "date"
    assert close.name == "close"


def test_deterministic(close):
    again, _ = data.synthetic_prices(n_bars=2520, kappa=0.06, seed=15)
    assert np.allclose(close.to_numpy(), again.to_numpy())


def test_seed_changes_path():
    a, _ = data.synthetic_prices(seed=1)
    b, _ = data.synthetic_prices(seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_mean_reversion_is_baked_in():
    """Stronger kappa => more negative one-step return autocorrelation (the real edge)."""
    weak, _ = data.synthetic_prices(kappa=0.01, sigma_step=0.011, seed=3)
    strong, _ = data.synthetic_prices(kappa=0.20, sigma_step=0.011, seed=3)
    r_weak = np.log(weak).diff().dropna()
    r_strong = np.log(strong).diff().dropna()
    ac_weak = r_weak.autocorr(lag=1)
    ac_strong = r_strong.autocorr(lag=1)
    assert ac_strong < ac_weak          # more reversion -> more negative serial correlation
    assert ac_strong < 0


def test_truth_reports_single_horizon(truth):
    assert truth.kappa > 0
    assert np.isclose(truth.horizon, 1.0 / truth.kappa)


def test_fetch_is_cache_only_offline():
    """No network in the offline core: a missing cache returns empty, never downloads."""
    s = data.fetch_prices("ZZZZ_NOPE", fetch=False)
    assert s.empty
