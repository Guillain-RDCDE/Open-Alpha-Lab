"""The synthetic series is deterministic; the MLP recovers the weak signal out-of-sample on the control
and finds none in the null; the in-sample Sharpe is hugely inflated versus walk-forward (the trap); the
shuffled-label control proves the in-sample number is noise-fitting; features are causal; costs behave.

Kept fast: short series, small net, few walk-forward folds (see conftest)."""

import numpy as np
import pandas as pd

from black_box import costs, data, extension, features, strategy
from conftest import FAST, SEED, WF


def test_series_deterministic_and_has_signal(signal_close, signal):
    df2, truth2 = data.synthetic_predictable(predictable_strength=0.015, n_bars=1200, seed=SEED)
    assert np.allclose(signal_close.to_numpy(), df2["close"].to_numpy())   # seeded → reproducible
    assert signal[1].has_signal
    df0, t0 = data.synthetic_predictable(predictable_strength=0.0, n_bars=1200, seed=SEED)
    assert not t0.has_signal


def test_features_are_causal(signal_close):
    """Every feature row for day t uses only data up to t-1: flipping the LAST close leaves all but the
    last feature row untouched."""
    X, _ = features.build_features(signal_close)
    c2 = signal_close.copy()
    c2.iloc[-1] *= 1.5
    X2, _ = features.build_features(c2)
    common = X.index.intersection(X2.index)[:-1]
    assert np.allclose(X.loc[common].to_numpy(), X2.loc[common].to_numpy())


def test_mlp_recovers_signal_oos_on_control(signal_close):
    """On the predictable control, walk-forward OOS Sharpe is positive — the net CAN learn the signal."""
    pos = strategy.walk_forward_predictions(signal_close, **WF)
    s = strategy.summary(strategy.book_returns(signal_close, pos, cost_bps=0.0))
    assert s["sharpe"] > 0.0


def test_oos_flat_on_null(null_close):
    """On the random-walk null there is nothing to learn — OOS Sharpe is near zero / small."""
    pos = strategy.walk_forward_predictions(null_close, **WF)
    s = strategy.summary(strategy.book_returns(null_close, pos, cost_bps=0.0))
    assert abs(s["sharpe"]) < 1.5


def test_insample_dwarfs_oos_the_trap(null_close):
    """THE TRAP: even on the null (no signal), in-sample Sharpe is much larger than walk-forward OOS,
    because the net memorises the labels it was trained on."""
    gap = extension.insample_vs_oos(null_close, cost_bps=10.0, **WF)
    is_sh = gap.loc["in_sample", "sharpe_gross"]
    oos_sh = gap.loc["walk_forward_oos", "sharpe_gross"]
    assert is_sh > oos_sh + 1.0          # in-sample edge is an artefact
    assert gap.loc["in_sample", "accuracy"] > gap.loc["walk_forward_oos", "accuracy"]


def test_shuffled_label_control_memorises_noise(null_close):
    """Shuffled targets carry NO information, yet the net's in-sample TRAINING ACCURACY stays well above
    the 0.5 coin-flip baseline — comparable to the true-label accuracy. Proof the in-sample fit measures
    memorisation capacity, not predictive skill (Bailey-Borwein-López de Prado-Zhu)."""
    tab = extension.shuffled_label_control(null_close, n_shuffles=3, **FAST)
    true_acc = tab.loc["true", "train_accuracy"]
    shuf_acc = tab.drop(index="true")["train_accuracy"]
    assert true_acc > 0.55                                 # the net fits the labels it was handed...
    assert shuf_acc.mean() > 0.55                          # ...just as well when they're pure noise


def test_cost_and_breakeven_behaviour(signal_close):
    pos = strategy.walk_forward_predictions(signal_close, **WF)
    cs = costs.cost_sweep(signal_close, pos)
    assert cs["sharpe"].iloc[0] >= cs["sharpe"].iloc[-1]   # higher cost ⇒ lower Sharpe
    assert costs.turnover(pos) > 0.0
    assert costs.breakeven_cost_bps(signal_close, pos) >= 0.0


def test_positions_are_pm_one(signal_close):
    pos = strategy.walk_forward_predictions(signal_close, **WF)
    assert set(np.unique(pos.to_numpy())).issubset({-1.0, 1.0})
