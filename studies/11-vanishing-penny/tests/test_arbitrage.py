"""The core claim of the machinery: both estimators recover the baked-in half-life."""

import numpy as np

from prediction_arb import arbitrage


def test_detects_a_healthy_number_of_episodes(gap):
    eps = arbitrage.detect_all(gap, open_threshold=0.03)
    # 48 markets over 6000 min at shock_rate 0.006 -> hundreds of openings
    assert len(eps) > 100
    assert all(abs(e.peak) >= 0.03 for e in eps)
    assert all(e.duration >= 1 for e in eps)


def test_empirical_time_to_half_recovers_truth(gap, truth):
    """The assumption-light median time-to-half lands on the baked-in 6.0 min."""
    eps = arbitrage.detect_all(gap, open_threshold=truth.open_threshold)
    h = arbitrage.time_to_half(eps)
    assert abs(h - truth.half_life_min) <= 1.2          # within one sample step of 6.0


def test_loglinear_fit_recovers_truth(gap, truth):
    """The pooled log-linear decay fit agrees with the median — decay is exponential."""
    eps = arbitrage.detect_all(gap, open_threshold=truth.open_threshold)
    h = arbitrage.fit_half_life(eps)
    assert abs(h - truth.half_life_min) <= 1.5


def test_half_life_scales_with_the_truth():
    """Bake a slower decay and the estimator tracks it — it's measuring the real thing."""
    from prediction_arb import data
    fast, _ = data.synthetic_markets(half_life_min=4.0, seed=3)
    slow, _ = data.synthetic_markets(half_life_min=12.0, seed=3)
    hf = arbitrage.time_to_half(arbitrage.detect_all(fast))
    hs = arbitrage.time_to_half(arbitrage.detect_all(slow))
    assert hf < hs
    assert abs(hf - 4.0) <= 1.2 and abs(hs - 12.0) <= 2.5


def test_summary_keys(gap):
    s = arbitrage.summary(arbitrage.detect_all(gap))
    for key in ("n_episodes", "half_life_median_min", "half_life_fit_min",
                "median_peak_penny", "frac_buy_both"):
        assert key in s
    assert 0.0 <= s["frac_buy_both"] <= 1.0


def test_empty_series_is_no_episodes():
    import pandas as pd
    s = pd.Series([0.0, 0.001, -0.002], index=pd.date_range("2024-01-01", periods=3, freq="1min"))
    assert arbitrage.detect_episodes(s, open_threshold=0.03) == []
    assert np.isnan(arbitrage.time_to_half([]))
