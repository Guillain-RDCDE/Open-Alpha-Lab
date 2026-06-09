"""The synthetic universe is deterministic and shaped right, and the predictability engine recovers the
baked trend on one tape and finds nothing on the null."""

import numpy as np

from trend_follow import data, trend


def test_deterministic_and_shaped(trend_panel):
    panel, truth = trend_panel
    panel2, _ = data.synthetic_panel(trend_strength=0.0006, seed=20)
    assert panel.shape == (truth.n_bars, truth.n_assets)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())
    assert truth.has_trend


def test_null_has_no_trend(null_panel):
    _, truth = null_panel
    assert not truth.has_trend


def test_trailing_and_vol_shapes(trend_panel):
    panel, _ = trend_panel
    tr = trend.trailing_return(panel, lookback=252)
    rv = trend.realized_vol(panel, window=63)
    assert tr.shape == panel.shape and rv.shape == panel.shape
    assert tr.iloc[:251].isna().all().all()
    assert (rv.dropna() > 0).all().all()


def test_predictability_recovers_trend(trend_panel):
    panel, _ = trend_panel
    pr = trend.predictability(panel)
    assert pr["pooled_t"] > 2.0        # past return predicts the next, significantly
    assert pr["hit_rate"] > 0.5
    assert pr["trend_present"]


def test_predictability_flat_on_null(null_panel):
    panel, _ = null_panel
    pr = trend.predictability(panel)
    assert abs(pr["pooled_t"]) < 2.0   # driftless noise: nothing to predict
    assert not pr["trend_present"]
