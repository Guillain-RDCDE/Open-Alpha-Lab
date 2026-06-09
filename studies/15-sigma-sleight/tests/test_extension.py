"""The beat-7 complement: a vol-adaptive band is NOT a relabel — its entry level moves with
realized vol. The baked-in checks: with no vol regime it collapses toward a static band (tiny
dispersion); with a regime it moves a lot; and the horse-race increment is an exact, unconstrained
difference of Sharpes (a time-varying rule genuinely can win or lose)."""

import numpy as np

from sigma_sleight import extension as ext


def test_constant_vol_threshold_barely_moves():
    """No regime (sigma_lo == sigma_hi): the vol ratio sits ~1, so the adaptive threshold has only
    sampling-noise dispersion — adaptivity has nothing to grab."""
    flat = ext.synthetic_regime_prices(sigma_lo=0.011, sigma_hi=0.011, seed=7)
    disp = ext.threshold_dispersion(flat, length=2, base_sigma=-1.0, gamma=0.5)
    assert disp < 3.0          # in RSI points: small, it barely breathes


def test_regime_vol_makes_threshold_move():
    """With a real lo/hi vol regime the entry level swings far more than under constant vol — proof
    it is a genuine time-varying rule, not an order-preserving relabel."""
    flat = ext.synthetic_regime_prices(sigma_lo=0.011, sigma_hi=0.011, seed=7)
    regime = ext.synthetic_regime_prices(sigma_lo=0.006, sigma_hi=0.024, seed=7)
    disp_flat = ext.threshold_dispersion(flat, length=2, base_sigma=-1.0, gamma=0.5)
    disp_regime = ext.threshold_dispersion(regime, length=2, base_sigma=-1.0, gamma=0.5)
    assert disp_regime > disp_flat * 2.0
    assert disp_regime > 3.0


def test_gamma_zero_is_a_static_band():
    """gamma=0 removes the vol-sensitivity: the threshold is a single constant (dispersion 0)."""
    regime = ext.synthetic_regime_prices(sigma_lo=0.006, sigma_hi=0.024, seed=1)
    assert ext.threshold_dispersion(regime, length=2, base_sigma=-1.0, gamma=0.0) < 1e-9


def test_positions_valid_and_shifted():
    regime = ext.synthetic_regime_prices(seed=2)
    pos = ext.vol_adaptive_positions(regime, length=2, base_sigma=-1.0, gamma=0.5)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}
    assert pos.iloc[0] == 0.0


def test_increment_is_exact_difference():
    regime = ext.synthetic_regime_prices(seed=3)
    out = ext.adaptive_vs_fixed(regime, length=2, gamma=0.5, cost_bps=1.0)
    assert np.isclose(out["increment_sharpe"],
                      out["adaptive"]["sharpe"] - out["fixed_best"]["sharpe"], atol=1e-12)
    # the threshold genuinely moved (regime tape), so this is not a relabel comparison
    assert out["threshold_dispersion"] > 1.0


def test_vol_ratio_centres_near_one():
    regime = ext.synthetic_regime_prices(seed=4)
    vr = ext.vol_ratio(regime).dropna()
    assert 0.7 < vr.median() < 1.4
    assert vr.max() > vr.min()
