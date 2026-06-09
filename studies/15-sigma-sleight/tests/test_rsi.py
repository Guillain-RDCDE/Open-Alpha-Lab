"""The RSI and the AdaptiveRSI transform behave exactly as the framework's arithmetic says:
RSI stays in [0,100], σ = logit·√(n−1)/2 is a strict monotone bijection, RSI(14)=70 lands on
+1.53σ, and the σ-zones compress toward 50 as length grows."""

import math

import numpy as np

from sigma_sleight import rsi as R


def test_rsi_in_bounds(close):
    for n in (2, 14, 50):
        v = R.wilder_rsi(close, n).dropna()
        assert (v >= 0).all() and (v <= 100).all()
        assert len(v) > 0


def test_rsi_warmup_is_nan(close):
    v = R.wilder_rsi(close, 14)
    assert v.iloc[:14].isna().all()
    assert v.iloc[14:].notna().any()


def test_sigma_roundtrip_is_identity():
    for n in (2, 14, 200):
        for x in (5.0, 30.0, 50.0, 70.0, 95.0):
            assert math.isclose(R.sigma_to_rsi(R.rsi_to_sigma(x, n), n), x, abs_tol=1e-6)


def test_rsi50_is_zero_sigma():
    for n in (2, 14, 200):
        assert abs(R.rsi_to_sigma(50.0, n)) < 1e-9


def test_marketing_landmark_rsi70_is_1p53_sigma():
    """The load-bearing claim: RSI(14)=70 maps to +1.53σ under logit·√(n−1)/2."""
    s = R.rsi_to_sigma(70.0, 14)
    assert math.isclose(s, 1.527, abs_tol=0.01)
    # and symmetric: RSI 30 is the mirror
    assert math.isclose(R.rsi_to_sigma(30.0, 14), -s, abs_tol=1e-9)


def test_sigma_map_is_strictly_increasing():
    grid = np.linspace(1.0, 99.0, 500)
    for n in (2, 14, 200):
        s = R.rsi_to_sigma(grid, n)
        assert np.all(np.diff(s) > 0)


def test_same_sigma_compresses_with_length():
    """A fixed σ landmark maps to an RSI level closer to 50 as the length grows."""
    levels = [R.sigma_to_rsi(1.0, n) for n in (2, 14, 50, 200)]
    assert levels[0] > levels[1] > levels[2] > levels[3] > 50.0


def test_zone_table_columns_and_compression():
    tab = R.zone_levels_table([2, 14, 200])
    assert list(tab.columns) == list(R.ZONE_SIGMA.keys())
    # every σ-zone's RSI level decreases monotonically down the lengths
    assert (tab.diff().dropna().to_numpy() < 0).all()


def test_rescaled_identity_when_lengths_match(close):
    a = R.wilder_rsi(close, 14)
    b = R.rescaled_rsi(close, 14, 14)
    valid = a.notna() & b.notna()
    assert np.allclose(a[valid].to_numpy(), b[valid].to_numpy(), atol=1e-6)
